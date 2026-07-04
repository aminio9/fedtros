import logging
import os
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch

logger = logging.getLogger("Environment")


class BlockchainIntrusionEnv(gym.Env):
    """
    MDP for Blockchain Intrusion Detection.

    - Environment: sample pool (optionally a shard of the full dataset)
    - State (s): traffic feature vector
    - Action (a): predicted label (discrete ID)
    - Reward (r): +1 if a == a_T, -1 otherwise
    - Transition: independent of action, follows sampled index order

    Designed to be:
      * Gymnasium-compatible
      * Reproducible via seeding
      * Usable in horizontal multi-agent / FL setups
        - Each client can load its own .pt file (simplest), or
        - Multiple clients can shard a common .pt file via indices/client_id.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        processed_data_path: str,
        steps_per_episode: int,
        *,
        device: torch.device | None = None,
        logger: logging.Logger | None = None,
        move_data_to_device: bool = False,
        indices: np.ndarray | None = None,
        client_id: int | None = None,
        num_clients: int | None = None,
        # NON-IID FIX: Accept global number of actions
        global_num_actions: int | None = None,
        reward_mode: str = "symmetric",
        reward_correct: float = 1.0,
        reward_wrong: float = -1.0,
        reward_weight_power: float = 0.5,
        reward_min_weight: float = 0.5,
        reward_max_weight: float = 3.0,
        reward_normalize_mean: bool = True,
    ) -> None:
        super().__init__()

        self.logger = logger or logging.getLogger("Environment")
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        target_device = self.device if move_data_to_device else torch.device("cpu")
        self._data_device = target_device
        self._move_data_to_device = move_data_to_device

        if not os.path.exists(processed_data_path):
            self.logger.error("Processed data not found at %s", processed_data_path)
            raise FileNotFoundError(f"Data not found: {processed_data_path}")

        try:
            data = torch.load(processed_data_path, map_location="cpu", weights_only=True)
            self.all_features_s: torch.Tensor = data["features"].to(
                device=target_device, dtype=torch.float32
            )
            self.all_labels_a_t: torch.Tensor = data["labels"].to(device=target_device).long()
        except Exception as exc:  # pragma: no cover - I/O error
            self.logger.error(
                "Failed to load data from %s: %s", processed_data_path, exc, exc_info=True
            )
            raise

        if self.all_features_s.ndim != 2:
            raise ValueError(
                f"'features' must be a 2D tensor [N, D], got shape {tuple(self.all_features_s.shape)}"
            )
        if self.all_labels_a_t.ndim != 1:
            raise ValueError(
                f"'labels' must be a 1D tensor [N], got shape {tuple(self.all_labels_a_t.shape)}"
            )
        if self.all_features_s.size(0) != self.all_labels_a_t.size(0):
            raise ValueError(
                "Mismatch between number of feature rows and labels: "
                f"{self.all_features_s.size(0)} vs {self.all_labels_a_t.size(0)}"
            )

        self.num_total_samples = self.all_features_s.shape[0]
        self.feature_dim = self.all_features_s.shape[1]
        self.steps_per_episode = int(steps_per_episode)
        self.reward_mode = str(reward_mode).lower()
        self.reward_correct = float(reward_correct)
        self.reward_wrong = float(reward_wrong)
        self.reward_weight_power = float(reward_weight_power)
        self.reward_min_weight = float(reward_min_weight)
        self.reward_max_weight = float(reward_max_weight)
        self.reward_normalize_mean = bool(reward_normalize_mean)
        self._reward_class_weights = torch.ones(1, dtype=torch.float32)
        if self.num_total_samples == 0:
            raise ValueError(f"No samples found in {processed_data_path}.")

        # ------------------------------------------------------------------
        # NON-IID FIX: Determine Action Space Size
        # ------------------------------------------------------------------
        if global_num_actions is not None:
            # Force the environment to use the global configuration
            self.num_actions_nt = int(global_num_actions)

            # Safety Check: Ensure local data doesn't contain labels outside global universe
            max_label = self.all_labels_a_t.max().item()
            if max_label >= self.num_actions_nt:
                raise ValueError(
                    f"Data contains label {max_label} which is >= global_num_actions ({self.num_actions_nt}). "
                    "Check your config or data mapping."
                )
        else:
            # Fallback: infer output size from the maximum class id, not the
            # number of unique local labels. Non-contiguous labels can appear in
            # non-IID shards.
            self.num_actions_nt = int(self.all_labels_a_t.max().item()) + 1

        # ------------------------------------------------------------------
        # Determine which indices belong to THIS env/agent
        # ------------------------------------------------------------------
        if indices is not None:
            idx_array = np.asarray(indices, dtype=np.int64)
            if idx_array.ndim != 1:
                raise ValueError("`indices` must be a 1D array of sample indices.")
            if idx_array.size == 0:
                raise ValueError("`indices` is empty; env would have no samples.")
            if idx_array.min() < 0 or idx_array.max() >= self.num_total_samples:
                raise ValueError(
                    f"Some values in `indices` are out of range [0, {self.num_total_samples - 1}]."
                )
            self._available_indices = idx_array
            self.logger.info("Environment using %d explicitly provided indices.", idx_array.size)
        elif client_id is not None and num_clients is not None:
            if num_clients <= 0:
                raise ValueError("`num_clients` must be > 0.")
            if not (0 <= client_id < num_clients):
                raise ValueError(f"`client_id` must be in [0, {num_clients - 1}], got {client_id}")

            all_indices = np.arange(self.num_total_samples, dtype=np.int64)
            shards = np.array_split(all_indices, num_clients)
            self._available_indices = shards[client_id]
            self.logger.info(
                "Client shard %d/%d: %d samples.",
                client_id,
                num_clients,
                self._available_indices.size,
            )
        else:
            # Simple single-client case: use all data
            self._available_indices = np.arange(self.num_total_samples, dtype=np.int64)
            self.logger.info("Environment using all %d samples.", self.num_total_samples)

        if self._available_indices.size == 0:
            raise ValueError("No samples available for this environment/agent.")

        self._configure_reward_weights()

        # RNG for episode sampling (decoupled from Gym's own RNG)
        self._rng = np.random.default_rng()

        # Episode state
        self.episode_indices: np.ndarray = np.empty(0, dtype=np.int64)
        self.current_step: int = 0

        # Gym spaces
        self.action_space = gym.spaces.Discrete(self.num_actions_nt)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.feature_dim,),
            dtype=np.float32,
        )

        self.logger.info(
            "Initialized BlockchainIntrusionEnv from %s | total=%d | client=%d | dim=%d | actions=%d | reward_mode=%s",
            os.path.basename(processed_data_path),
            self.num_total_samples,
            self._available_indices.size,
            self.feature_dim,
            self.num_actions_nt,
            self.reward_mode,
        )
        if move_data_to_device:
            self.logger.info("Environment tensors pinned to %s", target_device)


    def _configure_reward_weights(self) -> None:
        """Compute local class reward weights from the client's available data.

        The environment is a sampled classification bandit.  With an imbalanced
        label distribution, a plain +1/-1 reward makes the agent optimize mostly
        for the majority class.  The class-balanced reward keeps the average
        weight near one while giving rare labels a larger TD signal.
        """
        num_actions = max(int(self.num_actions_nt), 1)
        weights = torch.ones(num_actions, dtype=torch.float32)
        if self.reward_mode not in {"balanced", "class_balanced", "inverse_frequency"}:
            self._reward_class_weights = weights
            return

        labels_cpu = self.all_labels_a_t.detach().cpu()
        available = torch.as_tensor(self._available_indices, dtype=torch.long)
        local_labels = labels_cpu[available]
        counts = torch.bincount(local_labels.clamp_min(0), minlength=num_actions).float()[:num_actions]
        present = counts > 0
        if not bool(present.any().item()):
            self._reward_class_weights = weights
            return

        mean_present_count = counts[present].mean().clamp_min(1.0)
        raw = torch.ones_like(counts)
        raw[present] = (mean_present_count / counts[present].clamp_min(1.0)).pow(
            max(self.reward_weight_power, 0.0)
        )
        raw = torch.clamp(raw, min=self.reward_min_weight, max=self.reward_max_weight)
        if self.reward_normalize_mean:
            weighted_mean = (raw[present] * counts[present]).sum() / counts[present].sum().clamp_min(1.0)
            raw = raw / weighted_mean.clamp_min(1e-8)
            raw = torch.clamp(raw, min=self.reward_min_weight, max=self.reward_max_weight)

        self._reward_class_weights = raw
        weights_str = ", ".join(f"{idx}:{raw[idx].item():.3f}" for idx in range(num_actions))
        self.logger.info("Class-balanced reward weights | %s", weights_str)

    def _class_reward_weight(self, label: int) -> float:
        if label < 0 or label >= int(self._reward_class_weights.numel()):
            return 1.0
        return float(self._reward_class_weights[label].item())

    def get_reward_class_weights(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Return class weights for reward-aware supervised auxiliary loss."""
        target_device = torch.device(device) if device is not None else self.device
        return self._reward_class_weights.to(target_device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Resets the environment by sampling a fresh set of indices
        for the episode and returning the initial observation.
        """
        _ = options
        super().reset(seed=seed)
        if seed is not None:
            # Re-seed our own RNG for reproducibility if a seed is provided
            self._rng = np.random.default_rng(seed)

        self._sample_episode_indices()
        self.current_step = 0

        obs = self._get_current_state()
        info = {
            "true_label": self._get_current_true_label(),
            "index": int(self.episode_indices[self.current_step]),
        }
        return obs, info

    def step(self, action_a_t: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Executes one environment step.
        Returns: (next_state, reward, terminated, truncated, info)
        """
        # 1. Reward from current state.  In class-balanced mode the reward is
        # scaled by the inverse local class frequency so minority intrusion
        # classes are not drowned by the Normal class during Q-learning.
        true_label_a_t = self._get_current_true_label()
        true_label_int = int(true_label_a_t)
        action_int = int(action_a_t)
        class_weight = self._class_reward_weight(true_label_int)
        base_reward = self.reward_correct if action_int == true_label_int else self.reward_wrong
        reward_r: float = float(base_reward * class_weight)

        # 2. Info about *current* transition
        current_index = int(self.episode_indices[self.current_step])
        info: dict[str, Any] = {
            "true_label": true_label_int,
            "index": current_index,
            "reward_weight": float(class_weight),
            "reward_mode": self.reward_mode,
        }

        # 3. Advance time
        self.current_step += 1

        # 4. Check termination
        terminated = self.current_step >= self.steps_per_episode
        truncated = False  # No separate truncation logic for now

        # 5. Next observation
        if not terminated:
            next_obs = self._get_current_state()
        else:
            # Dummy obs (won't be used if the agent respects terminated=True)
            next_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

        return next_obs, reward_r, terminated, truncated, info

    def render(self) -> None:
        # This environment has no visual render; hook for logging/printing if desired.
        pass

    def close(self) -> None:
        # Nothing special to clean up; method present for API completeness.
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sample_episode_indices(self) -> None:
        """Sample indices for this episode using the env's RNG."""
        self.episode_indices = self._rng.choice(
            self._available_indices,
            size=self.steps_per_episode,
            replace=True,
        )

    def _get_current_state(self) -> np.ndarray:
        """Return feature vector for the current step as a NumPy array."""
        idx = int(self.episode_indices[self.current_step])
        return self.all_features_s[idx].cpu().numpy().astype(np.float32)

    def _get_current_true_label(self) -> int:
        """Return true label for the current step."""
        idx = int(self.episode_indices[self.current_step])
        return int(self.all_labels_a_t[idx].item())
