"""UR7E joint-position contract for OpenPI VLASh training and serving."""

import dataclasses

import einops
import numpy as np

from openpi import transforms


def _image(value):
    image = np.asarray(value)
    if np.issubdtype(image.dtype, np.floating):
        image = (image * 255).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Inputs:
    def __call__(self, data):
        base = _image(data["images"]["cam_high"])
        result = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": _image(data["images"]["cam_low"]),
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
        }
        for key in ("actions", "action_loss_mask", "prompt"):
            if key in data:
                result[key] = data[key]
        return result


@dataclasses.dataclass(frozen=True)
class Outputs:
    def __call__(self, data):
        return {"state": np.array(data["state"], copy=True), "actions": np.array(data["actions"][..., :7], copy=True)}


def data_transforms(_model):
    mask = transforms.make_bool_mask(6, -1)
    return transforms.Group(
        inputs=[Inputs(), transforms.DeltaActions(mask)],
        outputs=[Outputs(), transforms.AbsoluteActions(mask)],
    )
