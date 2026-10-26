from unittest import mock

import numpy as np

from openpi.training.vlash_dataset import DelayedActionDataset


class Dataset:
    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return {"observation.state": np.full(7, -1, dtype=np.float32),
                "action": np.tile(np.arange(7, dtype=np.float32)[:, None], (1, 7)),
                "action_is_pad": np.array([False] * 5 + [True] * 2),
                "image": np.array([123])}


def test_future_state_and_shifted_labels():
    dataset = DelayedActionDataset(Dataset(), 4, 3, ("action",))
    with mock.patch("numpy.random.randint", return_value=3):
        sample = dataset[0]
    np.testing.assert_equal(sample["observation.state"], np.full(7, 2))
    np.testing.assert_equal(sample["action"][:, 0], [3, 4, 5, 6])
    np.testing.assert_equal(sample["action_loss_mask"], [True, True, False, False])
    np.testing.assert_equal(sample["image"], [123])


def test_zero_delay_retains_measured_state():
    with mock.patch("numpy.random.randint", return_value=0):
        sample = DelayedActionDataset(Dataset(), 4, 3, ("action",))[0]
    np.testing.assert_equal(sample["observation.state"], np.full(7, -1))
    np.testing.assert_equal(sample["action"][:, 0], [0, 1, 2, 3])
