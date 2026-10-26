"""Training-time future-state awareness for OpenPI pi05 on UR7E.

Adapted from VLASh's async_lora.yaml, VLASHDataset and PI05SuffixEmbedder.
This uses OpenPI JAX/LoRA; it does not load VLASh's PyTorch checkpoints.
"""

import dataclasses

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import vlash_ur7e_policy
from openpi.training import config
from openpi.training import weight_loaders


@dataclasses.dataclass(frozen=True)
class VlashWeightLoader:
    params_path: str

    def load(self, params):
        from openpi.models import model
        from openpi.shared import download

        import numpy as np

        loaded = model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return weight_loaders._merge_params(
            loaded, params, missing_regex=r".*lora.*|state_(proj|mlp_in|mlp_out)/.*"
        )


def get_configs():
    model = pi0_config.Pi0Config(
        pi05=True, action_horizon=50, discrete_state_input=False, vlash_state_cond=True,
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
    )
    return [config.TrainConfig(
        name="pi05_ur7e_vlash_lora",
        model=model,
        data=config.SimpleDataConfig(
            repo_id="dataset0924",
            data_transforms=vlash_ur7e_policy.data_transforms,
            model_transforms=config.ModelTransformFactory(default_prompt="UR7E pick and place"),
            base_config=config.DataConfig(
                repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                    "images": {"cam_high": "observation.images.camera_0", "cam_low": "observation.images.camera_1"},
                    "state": "observation.state", "actions": "action", "prompt": "prompt",
                    "action_loss_mask": "action_loss_mask",
                })]),
                action_sequence_keys=("action",), prompt_from_task=True, vlash_max_delay_steps=10,
            ),
        ),
        weight_loader=VlashWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=model.get_freeze_filter(), ema_decay=None, batch_size=4,
        num_train_steps=30_000, wandb_enabled=False,
        policy_metadata={"deployment": "vlash", "action_horizon": 50, "state_shape": [7], "action_shape": [7],
                         "control_contract": {"control_mode": "joint", "state_source": "future_absolute_action"}},
    )]
