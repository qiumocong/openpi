import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR7EInputs(transforms.DataTransformFn):
    """Maps UR7E joint-space data and two cameras to the OpenPI inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["images"]["cam_high"])
        wrist_image = _parse_image(data["images"]["cam_low"])
        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": self.model_type == _model.ModelType.PI0_FAST,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class UR7EOutputs(transforms.DataTransformFn):
    """Returns the seven UR7E joint/gripper command dimensions."""

    def __call__(self, data: dict) -> dict:
        # Keep state in the output tree: the data-side AbsoluteActions
        # transform runs after this transform and needs it to restore the six
        # joint dimensions from deltas to absolute UR targets.
        return {
            "state": np.asarray(data["state"]),
            "actions": np.asarray(data["actions"][..., :7]),
        }
