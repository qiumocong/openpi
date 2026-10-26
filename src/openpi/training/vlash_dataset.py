"""VLASh random-delay targets, adapted from mit-han-lab/vlash (Apache-2.0)."""

import numpy as np


class DelayedActionDataset:
    """Keep images at t; use a[t+d-1] as state and labels a[t+d:t+d+H].

    LeRobot supplies H+N actions already clamped to the current episode and
    their padding mask, so neither targets nor future state cross episodes.
    """

    def __init__(self, dataset, horizon: int, max_delay_steps: int, action_keys):
        if max_delay_steps < 0 or horizon < 1 or tuple(action_keys) != ("action",):
            raise ValueError("VLASh currently requires a positive horizon and the single absolute 'action' key")
        self.dataset = dataset
        self.horizon = horizon
        self.max_delay_steps = max_delay_steps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        actions = np.array(sample["action"], dtype=np.float32, copy=True)
        state = np.asarray(sample["observation.state"], dtype=np.float32)
        if actions.shape != (self.horizon + self.max_delay_steps, state.shape[-1]) or state.ndim != 1:
            raise ValueError("VLASh requires matching absolute action/state coordinates and H+N action frames")
        if "action_is_pad" not in sample:
            raise ValueError("VLASh requires LeRobot action_is_pad to mask episode-end targets")
        padding = np.asarray(sample["action_is_pad"], dtype=bool)
        delay = int(np.random.randint(self.max_delay_steps + 1))
        if delay:
            sample["observation.state"] = actions[delay - 1].copy()
        sample["action"] = actions[delay : delay + self.horizon].copy()
        sample["action_is_pad"] = padding[delay : delay + self.horizon].copy()
        sample["action_loss_mask"] = ~sample["action_is_pad"]
        return sample
