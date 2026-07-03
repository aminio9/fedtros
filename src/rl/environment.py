import logging
import os
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch

<<<<<<< HEAD
=======
from src.utils.imbalance import class_weights_from_config, sample_weights_from_class_weights

>>>>>>> ea28efe (Initial commit with updated source code)
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
<<<<<<< HEAD
=======
        reward_correct: float = 1.0,
        reward_incorrect: float = -1.0,
        class_balanced_rewards: bool = False,
        class_balance_power: float = 1.0,
        imbalance_cfg: object | None = None,
>>>>>>> ea28efe (Initial commit with updated source code)
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

<<<<<<< HEAD
=======
        self.reward_correct = float(reward_correct)
        self.reward_incorrect = float(reward_incorrect)
        self.class_balanced_rewards = bool(class_balanced_rewards)
        self.class_balance_power = float(class_balance_power)
        self._imbalance_cfg = imbalance_cfg
        self._weight_negative_reward = bool(
            getattr(imbalance_cfg, "weight_negative_reward", False)
            if imbalance_cfg is not None
            else False
        )
        self._class_reward_weights = self._build_class_reward_weights()

>>>>>>> ea28efe (Initial commit with updated source code)
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

<<<<<<< HEAD
=======
        self._sampling_probabilities = self._build_sampling_probabilities()

>>>>>>> ea28efe (Initial commit with updated source code)
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
            "Initialized BlockchainIntrusionEnv from %s | total=%d | client=%d | dim=%d | actions=%d",
            os.path.basename(processed_data_path),
            self.num_total_samples,
            self._available_indices.size,
            self.feature_dim,
            self.num_actions_nt,
        )
        if move_data_to_device:
            self.logger.info("Environment tensors pinned to %s", target_device)

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
        # 1. Reward from current state
        true_label_a_t = self._get_current_true_label()
<<<<<<< HEAD
        reward_r: float = 1.0 if int(action_a_t) == int(true_label_a_t) else -1.0
=======
        class_weight = float(self._class_reward_weights[int(true_label_a_t)].item())
        if int(action_a_t) == int(true_label_a_t):
            reward_r = self.reward_correct * class_weight
        else:
            if self._weight_negative_reward:
                reward_r = self.reward_incorrect * class_weight
            else:
                reward_r = self.reward_incorrect
>>>>>>> ea28efe (Initial commit with updated source code)

        # 2. Info about *current* transition
        current_index = int(self.episode_indices[self.current_step])
        info: dict[str, Any] = {
            "true_label": int(true_label_a_t),
            "index": current_index,
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
<<<<<<< HEAD
        )

=======
            p=self._sampling_probabilities,
        )

    def _build_class_reward_weights(self) -> torch.Tensor:
        """Return inverse-frequency weights for correct rewards when enabled."""
        weights = torch.ones(self.num_actions_nt, dtype=torch.float32, device=self._data_device)
        if self._imbalance_cfg is not None and bool(getattr(self._imbalance_cfg, "enabled", False)):
            cfg_weights = class_weights_from_config(
                self.all_labels_a_t.detach().cpu(),
                num_classes=self.num_actions_nt,
                cfg=self._imbalance_cfg,
            )
            if bool(getattr(self._imbalance_cfg, "weighted_reward", False)):
                weights = cfg_weights.to(device=self._data_device)
            if bool(getattr(self._imbalance_cfg, "log_weights", True)):
                self.logger.info(
                    "Class imbalance config | weights=%s | weighted_reward=%s | "
                    "weight_negative=%s",
                    [round(float(v), 6) for v in cfg_weights.tolist()],
                    bool(getattr(self._imbalance_cfg, "weighted_reward", False)),
                    self._weight_negative_reward,
                )
            return weights
        if not self.class_balanced_rewards:
            return weights

        labels = self.all_labels_a_t.detach().long()
        valid = (labels >= 0) & (labels < self.num_actions_nt)
        if not bool(valid.any()):
            return weights

        counts = torch.bincount(labels[valid].cpu(), minlength=self.num_actions_nt).float()
        present = counts > 0
        if not bool(present.any()):
            return weights

        mean_count = counts[present].mean()
        balanced = torch.ones_like(counts)
        balanced[present] = (mean_count / counts[present]).pow(self.class_balance_power)
        balanced[present] = balanced[present] / balanced[present].mean().clamp_min(1e-12)
        return balanced.to(device=self._data_device)

    def _build_sampling_probabilities(self) -> np.ndarray | None:
        if self._imbalance_cfg is None or not bool(getattr(self._imbalance_cfg, "enabled", False)):
            return None
        if not bool(getattr(self._imbalance_cfg, "class_balanced_sampling", False)):
            return None
        class_weights = class_weights_from_config(
            self.all_labels_a_t.detach().cpu(),
            num_classes=self.num_actions_nt,
            cfg=self._imbalance_cfg,
        )
        sample_weights = sample_weights_from_class_weights(
            self.all_labels_a_t.detach().cpu(),
            class_weights,
        )
        available_weights = sample_weights[self._available_indices].numpy().astype(np.float64)
        if not np.isfinite(available_weights).all() or available_weights.sum() <= 0.0:
            self.logger.warning("Invalid class-balanced sampling weights; using uniform sampling.")
            return None
        probabilities = available_weights / available_weights.sum()
        self.logger.info("Class-balanced environment sampling enabled.")
        return probabilities

>>>>>>> ea28efe (Initial commit with updated source code)
    def _get_current_state(self) -> np.ndarray:
        """Return feature vector for the current step as a NumPy array."""
        idx = int(self.episode_indices[self.current_step])
        return self.all_features_s[idx].cpu().numpy().astype(np.float32)

    def _get_current_true_label(self) -> int:
        """Return true label for the current step."""
        idx = int(self.episode_indices[self.current_step])
        return int(self.all_labels_a_t[idx].item())
