from __future__ import annotations

import dataclasses
import logging
import pathlib
import platform
import sys

import etils.epath as epath
import jax
import tyro

from openpi.training import checkpoints as checkpoint_utils
from openpi.training import config as config_lib
from openpi.training import sharding

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train as train_lib  # noqa: E402


@dataclasses.dataclass
class Args:
    config_name: str
    exp_name: str
    checkpoint_base_dir: str = "/storage/nobackup/yiduo/acotvla_checkpoints"
    step: int = 0
    overwrite: bool = False
    seed: int | None = None


class StaticDataLoader:
    def __init__(self, data_config):
        self._data_config = data_config

    def data_config(self):
        return self._data_config


def main(args: Args) -> None:
    train_lib.init_logging()
    logging.info("Running on: %s", platform.node())
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    config = config_lib.get_config(args.config_name)
    config = dataclasses.replace(
        config,
        exp_name=args.exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        overwrite=args.overwrite,
        resume=False,
        wandb_enabled=False,
    )
    if args.seed is not None:
        config = dataclasses.replace(config, seed=args.seed)

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise ValueError(
            f"Config {args.config_name} did not resolve any norm stats. Compute them first before exporting a smoke checkpoint."
        )

    checkpoint_manager, _ = checkpoint_utils.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=False,
    )

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)

    train_state, _ = train_lib.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    logging.info("Initialized state for %s", args.config_name)

    data_loader = StaticDataLoader(data_config)
    checkpoint_utils.save_state(checkpoint_manager, train_state, data_loader, args.step)
    checkpoint_manager.wait_until_finished()

    output_dir = config.checkpoint_dir / str(args.step)
    logging.info("Exported step-%s checkpoint to %s", args.step, output_dir)
    print(output_dir)


if __name__ == "__main__":
    tyro.cli(main)
