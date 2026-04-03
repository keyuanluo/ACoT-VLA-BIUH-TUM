from __future__ import annotations

import dataclasses
import pathlib
import sys

import tyro

from openpi.training import config as config_lib
from openpi.training import weight_loaders

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train as train_lib  # noqa: E402

DEFAULT_TASKS = [
    "clean_the_desktop_addition",
    "clean_the_desktop_part_1",
    "clean_the_desktop_part_2",
    "hold_pot",
    "open_door",
    "place_block_into_box",
    "pour_workpiece",
    "scoop_popcorn",
    "scoop_popcorn_part_2",
    "sorting_packages_part_1",
    "sorting_packages_part_2",
]


@dataclasses.dataclass
class Args:
    exp_name: str
    dataset_root: str = "/storage/nobackup/yiduo/agibot_challenge_2026/Reasoning2Action-Sim/dataset_without_depth"
    tasks: list[str] = dataclasses.field(default_factory=lambda: DEFAULT_TASKS.copy())
    assets_dir: str = "/storage/home/yiduo/ACoT-VLA/assets/competition"
    checkpoint_base_dir: str = "/storage/nobackup/yiduo/acotvla_without_depth_checkpoints"
    weight_path: str = "/storage/nobackup/yiduo/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
    num_train_steps: int = 1000
    batch_size: int = 16
    num_workers: int = 8
    log_interval: int = 20
    val_split_ratio: float = 0.01
    val_interval: int = 100
    val_num_batches: int | None = 64
    save_interval: int = 500
    keep_period: int | None = 500
    wandb_enabled: bool = False
    overwrite: bool = False
    resume: bool = False
    fsdp_devices: int = 1
    seed: int = 42


def _validate_task_roots(dataset_root: pathlib.Path, tasks: list[str]) -> list[str]:
    repo_ids: list[str] = []
    for task in tasks:
        task_root = dataset_root / task
        for child in ["data", "meta", "videos"]:
            if not (task_root / child).exists():
                raise FileNotFoundError(f"{task_root / child} is missing. Extract dataset archives before training.")
        repo_ids.append(str(task_root))
    return repo_ids


def build_config(args: Args):
    dataset_root = pathlib.Path(args.dataset_root)
    repo_ids = _validate_task_roots(dataset_root, args.tasks)
    if "pi05_base" in args.weight_path:
        init_tag = "pi05_base"
    elif "pi05_libero" in args.weight_path:
        init_tag = "pi05_libero"
    else:
        init_tag = pathlib.Path(args.weight_path).parent.name.replace("-", "_")

    base = config_lib.get_config("acot_icra_simulation_challenge_reasoning_to_action")
    data = dataclasses.replace(
        base.data,
        repo_id=repo_ids,
        assets=config_lib.AssetsConfig(
            assets_dir=args.assets_dir,
            asset_id=list(args.tasks),
        ),
    )

    return dataclasses.replace(
        base,
        name=f"acot_icra_simulation_challenge_reasoning_to_action_without_depth_local_{init_tag}",
        exp_name=args.exp_name,
        data=data,
        weight_loader=weight_loaders.ACOTCheckpointWeightLoader(args.weight_path),
        checkpoint_base_dir=args.checkpoint_base_dir,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        val_split_ratio=args.val_split_ratio,
        val_interval=args.val_interval,
        val_num_batches=args.val_num_batches,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        wandb_enabled=args.wandb_enabled,
        overwrite=args.overwrite,
        resume=args.resume,
        fsdp_devices=args.fsdp_devices,
        seed=args.seed,
    )


def main(args: Args) -> None:
    config = build_config(args)
    train_lib.main(config)


if __name__ == "__main__":
    main(tyro.cli(Args))
