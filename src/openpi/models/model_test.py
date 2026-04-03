from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import acot_vla
from openpi.models import pi0
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_observation_from_dict_keeps_depth_separate():
    rgb_image = jnp.full((2, 4, 5, 3), 255, dtype=jnp.uint8)
    depth_image = jnp.arange(2 * 4 * 5, dtype=jnp.uint16).reshape(2, 4, 5)

    observation = _model.Observation.from_dict(
        {
            "image": {"base_0_rgb": rgb_image},
            "image_mask": {"base_0_rgb": jnp.array([True, True])},
            "depth_image": {"base_0_depth": depth_image},
            "state": jnp.ones((2, 8), dtype=jnp.float32),
        }
    )

    assert observation.images["base_0_rgb"].dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(observation.images["base_0_rgb"]), 1.0)

    assert observation.depth_images is not None
    assert observation.depth_images["base_0_depth"].dtype == jnp.float32
    assert observation.depth_images["base_0_depth"].shape == (2, 4, 5, 1)
    np.testing.assert_array_equal(
        np.asarray(observation.depth_images["base_0_depth"][..., 0]),
        np.asarray(depth_image.astype(jnp.float32)),
    )
    assert observation.depth_image_masks is None

    as_dict = observation.to_dict()
    assert "depth_image" in as_dict
    assert "base_0_depth" in as_dict["depth_image"]
    assert "base_0_depth" not in as_dict["image"]


def test_preprocess_observation_preserves_depth_branch():
    rgb = jnp.ones((1, 8, 8, 3), dtype=jnp.float32)
    depth = jnp.full((1, 8, 8, 1), 123.0, dtype=jnp.float32)
    observation = _model.Observation(
        images={
            "base_0_rgb": rgb,
            "left_wrist_0_rgb": rgb,
            "right_wrist_0_rgb": rgb,
        },
        image_masks={
            "base_0_rgb": jnp.array([True]),
            "left_wrist_0_rgb": jnp.array([True]),
            "right_wrist_0_rgb": jnp.array([True]),
        },
        depth_images={"base_0_depth": depth},
        depth_image_masks=None,
        state=jnp.ones((1, 8), dtype=jnp.float32),
    )

    processed = _model.preprocess_observation(None, observation, train=False, image_resolution=(8, 8))

    assert processed.depth_images is not None
    np.testing.assert_array_equal(np.asarray(processed.depth_images["base_0_depth"]), np.asarray(depth))
    assert processed.depth_image_masks is not None
    np.testing.assert_array_equal(np.asarray(processed.depth_image_masks["base_0_depth"]), np.array([True]))


def test_acot_fake_obs_declares_depth_branch():
    config = acot_vla.ACOTConfig()
    observation = config.fake_obs(batch_size=2)

    assert set(observation.images) == set(_model.IMAGE_KEYS)
    assert set(observation.depth_images) == set(_model.DEPTH_IMAGE_KEYS)
    assert set(observation.depth_image_masks) == set(_model.DEPTH_IMAGE_KEYS)

    for key in _model.DEPTH_IMAGE_KEYS:
        assert observation.depth_images[key].shape == (2, *_model.IMAGE_RESOLUTION, 1)
        assert observation.depth_image_masks[key].shape == (2,)


def test_depth_expert_outputs_expected_token_shapes():
    depth_expert = acot_vla.DepthExpert(
        camera_names=_model.DEPTH_IMAGE_KEYS,
        patch_size=14,
        embed_dim=32,
        encoder_depth=1,
        num_heads=4,
        tokens_per_view=4,
        reasoner_dim=48,
        expert_dim=64,
        rngs=nnx.Rngs(jax.random.key(0)),
    )

    depth_image = jnp.ones((2, 224, 224, 1), dtype=jnp.float32)
    prepared = depth_expert._prepare_single_view(depth_image)
    patch_grid = depth_expert.patch_embed(prepared)
    assert patch_grid.shape == (2, 16, 16, 32)

    observation = _model.Observation(
        images={key: jnp.ones((2, 224, 224, 3), dtype=jnp.float32) for key in _model.IMAGE_KEYS},
        image_masks={key: jnp.ones((2,), dtype=jnp.bool_) for key in _model.IMAGE_KEYS},
        depth_images={key: depth_image for key in _model.DEPTH_IMAGE_KEYS},
        depth_image_masks={key: jnp.ones((2,), dtype=jnp.bool_) for key in _model.DEPTH_IMAGE_KEYS},
        state=jnp.ones((2, 32), dtype=jnp.float32),
    )

    reasoner_tokens, expert_tokens = depth_expert(observation)
    assert reasoner_tokens is not None
    assert expert_tokens is not None
    assert reasoner_tokens.shape == (2, 12, 48)
    assert expert_tokens.shape == (2, 12, 64)


def test_depth_expert_preserves_unit_range_float_depth():
    depth_expert = acot_vla.DepthExpert(
        camera_names=_model.DEPTH_IMAGE_KEYS,
        patch_size=14,
        embed_dim=16,
        encoder_depth=1,
        num_heads=4,
        tokens_per_view=4,
        reasoner_dim=24,
        expert_dim=32,
        rngs=nnx.Rngs(jax.random.key(1)),
    )

    depth_image = jnp.full((1, 224, 224, 1), 0.5, dtype=jnp.float32)
    prepared = depth_expert._prepare_single_view(depth_image)

    np.testing.assert_allclose(np.asarray(prepared[..., 0]), 0.5)
    np.testing.assert_array_equal(np.asarray(prepared[..., 1]), 1.0)


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
