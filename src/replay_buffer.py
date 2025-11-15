import torch
import numpy as np
import random
from collections import deque, namedtuple
from typing import Tuple

# Define the structure of an experience
Experience = namedtuple(
    'Experience',
    (
        'state_s',
        'action_a_t',
        'reward_r',
        'next_state_s',
        'done',
        'true_action_a_t', # Store true label for PriorNet training
    ),
)

class ExperienceReplayBuffer:
    """A simple FIFO experience replay buffer."""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_s, action_a_t, reward_r, next_state_s, done, true_action_a_t):
        """Save an experience."""
        # Ensure data is on CPU and in basic types for storage
        self.buffer.append(
            Experience(
                np.asarray(state_s),
                int(action_a_t),
                float(reward_r),
                np.asarray(next_state_s),
                float(done),
                int(true_action_a_t)
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch
        states_s, actions_a_t, rewards_r, next_states_s, dones, true_actions_a_t = zip(*batch)
        
        # Convert to Tensors on the correct device
        states_s = torch.tensor(np.array(states_s), dtype=torch.float32, device=device)
        actions_a_t = torch.tensor(actions_a_t, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_r = torch.tensor(rewards_r, dtype=torch.float32, device=device).unsqueeze(1)
        next_states_s = torch.tensor(np.array(next_states_s), dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        true_actions_a_t = torch.tensor(true_actions_a_t, dtype=torch.int64, device=device).unsqueeze(1)
            
        return states_s, actions_a_t, rewards_r, next_states_s, dones, true_actions_a_t

    def __len__(self) -> int:
        return len(self.buffer)
