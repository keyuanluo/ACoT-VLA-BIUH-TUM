from __future__ import annotations

import dataclasses
import functools
import logging
import pathlib
import platform
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import model as model_lib
from openpi.training import config as config_lib
from openpi.training import data_loader as data_loader_lib
from openpi.training import sharding
from openpi.training import utils as training_utils

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train as train_lib  # noqa: E402


@dataclasses.dataclass
class Args:
    config_name: str = "acot_icra_simulation_challenge_reasoning_to_action_clean_depth_pi05libero_continue_45000"
    per_device_batch_size: int = 1
    num_workers: int = 0
    fsdp_devices: int = 1
    dataset_split: str = "train"
    num_batches: int = 1
    seed: int | None = None
    skip_norm_stats: bool = False
    init_state: bool = False
    train_step: bool = False


def _summarize_observation(observation: model_lib.Observation) -> None:
    print("rgb_keys=", sorted(observation.images.keys()))
    print("depth_keys=", sorted(observation.depth_images.keys()) if observation.depth_images is not None else [])
    print("rgb_shapes=", {k: tuple(v.shape) for k, v in observation.images.items()})
    if observation.depth_images is not None:
        print("depth_shapes=", {k: tuple(v.shape) for k, v in observation.depth_images.items()})
    if observation.depth_image_masks is not None:
        print("depth_mask_shapes=", {k: tuple(v.shape) for k, v in observation.depth_image_masks.items()})


def main(args: Args) -> None:
    train_lib.init_logging()
    logging.info("Running on: %s", platform.node())
    logging.info("JAX backend: %s", jax.default_backend())
    logging.info("JAX devices: %s", jax.devices())
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    if jax.default_backend() == "cpu" and (args.init_state or args.train_step):
        logging.warning("CPU backend detected. State init / train step for the full ACOT model may be slow or OOM.")

    batch_size = args.per_device_batch_size * jax.device_count()
    config = config_lib.get_config(args.config_name)
    config = dataclasses.replace(
        config,
        batch_size=batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        val_split_ratio=0.0,
        wandb_enabled=False,
        resume=False,
        overwrite=False,
    )
    if args.seed is not None:
        config = dataclasses.replace(config, seed=args.seed)

    logging.info(
        "Resolved config=%s batch_size=%s per_device_batch_size=%s fsdp_devices=%s checkpoint=%s",
        config.name,
        config.batch_size,
        args.per_device_batch_size,
        config.fsdp_devices,
        config.weight_loader.params_path,
    )

    mesh = sharding.make_mesh(config.fsdp_devices)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    loader = data_loader_lib.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=args.num_batches,
        skip_norm_stats=args.skip_norm_stats,
        dataset_split=args.dataset_split,
    )
    batch = next(iter(loader))
    observation = batch[0]
    logging.info("Loaded batch:\n%s", training_utils.array_tree_to_info(batch))
    _summarize_observation(observation)

    if not args.init_state and not args.train_step:
        return

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    train_state, train_state_sharding = train_lib.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state.step)
    logging.info("Initialized train state.")
    logging.info("Total number of parameters: %s", f"{training_utils.count_parameters(train_state.params):,}")

    if not args.train_step:
        return

    if config.model.model_type in (model_lib.ModelType.ACOT_VLA_PI05, model_lib.ModelType.ACOT_VLA_PI0):
        ptrain_step = jax.jit(
            functools.partial(train_lib.acot_train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(train_lib.train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    with sharding.set_mesh(mesh):
        train_state, info = ptrain_step(train_rng, train_state, batch)
    info = jax.device_get(jax.tree.map(lambda x: np.asarray(x).item() if np.asarray(x).shape == () else np.asarray(x), info))
    jax.block_until_ready(train_state.step)
    print("train_step_info=", info)


if __name__ == "__main__":
    main(tyro.cli(Args))
