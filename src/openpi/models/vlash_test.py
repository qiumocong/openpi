import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import pi0_config
from openpi.training import config


def test_vlash_config_rejects_discrete_state():
    with pytest.raises(ValueError):
        pi0_config.Pi0Config(pi05=True, vlash_state_cond=True)


def test_state_conditioning_and_masked_flow_loss_shapes():
    cfg = pi0_config.Pi0Config(
        pi05=True, vlash_state_cond=True, discrete_state_input=False,
        paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4,
    )

    def trace():
        model = cfg.create(jax.random.key(0))
        obs = dataclasses.replace(cfg.fake_obs(), action_loss_mask=jnp.array([[True, True, False, False]]))
        tokens, _, _, cond = model.embed_suffix(obs, cfg.fake_act(), jnp.ones(1))
        loss = model.compute_loss(jax.random.key(1), obs, cfg.fake_act())
        return tokens, cond, loss

    tokens, cond, loss = nnx.eval_shape(trace)
    assert tokens.shape[:2] == (1, 4)
    assert cond.shape[0] == 1
    assert loss.shape == (1, 4)


def test_new_state_weights_are_trainable():
    cfg = config.get_config("pi05_ur7e_vlash_lora")
    model = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    trainable = nnx.state(model, cfg.trainable_filter).flat_state()
    assert any("state_proj" in path for path in trainable)
    assert any("state_mlp_in" in path for path in trainable)
    assert any("state_mlp_out" in path for path in trainable)
