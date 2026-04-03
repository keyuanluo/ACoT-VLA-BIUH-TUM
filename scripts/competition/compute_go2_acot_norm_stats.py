from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pandas as pd
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.transforms as transforms


STATE_INDICES_159 = tuple(list(range(30, 44)) + [0, 1] + list(range(75, 80)))
ACTION_DIM = 32
COARSE_ACTION_HORIZON = 30
ACTION_HORIZON = 30
COARSE_SHIFT = 2
ACTION_SHIFT = 1
ACTION_CHUNK_SIZE = max(COARSE_ACTION_HORIZON * COARSE_SHIFT, ACTION_HORIZON * ACTION_SHIFT)
STATE_MASK = np.asarray(transforms.make_bool_mask(-14, 2, 4, -1, 11), dtype=bool)
ACTION_MASK = np.asarray(transforms.make_bool_mask(-16, 4, -1, 11), dtype=bool)
DELTA_MASK = np.asarray(transforms.make_bool_mask(14, -18), dtype=bool)


@dataclasses.dataclass
class Args:
    dataset_root: str = "/storage/nobackup/yiduo/agibot_challenge_2026/Reasoning2Action-Sim/clean_the_desktop_addition"
    output_dir: str = "/storage/home/yiduo/ACoT-VLA/assets/competition/clean_the_desktop_addition"
    max_episodes: int | None = None


def _load_episode_arrays(parquet_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path, columns=["observation.state", "action"])
    states = np.stack(df["observation.state"].map(lambda x: np.asarray(x, dtype=np.float32)).to_list())
    actions = np.stack(df["action"].map(lambda x: np.asarray(x, dtype=np.float32)).to_list())
    if states.shape[1] != 159:
        raise ValueError(f"Expected 159-dim state in {parquet_path}, got {states.shape}")
    if actions.shape[1] != 40:
        raise ValueError(f"Expected 40-dim action in {parquet_path}, got {actions.shape}")
    return states, actions


def _transform_state(raw_state: np.ndarray) -> np.ndarray:
    state = raw_state[np.asarray(STATE_INDICES_159)]
    state = transforms.pad_to_dim(state, ACTION_DIM).astype(np.float32, copy=False)
    state[STATE_MASK[: state.shape[-1]]] = 0
    return state


def _build_action_window(actions: np.ndarray, start_idx: int) -> np.ndarray:
    indices = np.arange(start_idx, start_idx + ACTION_CHUNK_SIZE)
    indices = np.clip(indices, 0, len(actions) - 1)
    return actions[indices]


def _normalize_action_chunk(chunk: np.ndarray, state_delta: np.ndarray) -> np.ndarray:
    chunk = chunk.copy()
    chunk[:, ACTION_MASK[: chunk.shape[-1]]] = 0.0
    chunk = transforms.pad_to_dim(chunk, ACTION_DIM).astype(np.float32, copy=False)
    chunk[:, :ACTION_DIM] -= state_delta[None, :ACTION_DIM]
    return chunk


def _transform_actions(raw_window: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw_actions = np.column_stack((raw_window[:, 16:30], raw_window[:, 0:2], raw_window[:, 33:38])).astype(np.float32)

    coarse_actions = raw_actions[: (COARSE_ACTION_HORIZON - 1) * COARSE_SHIFT + 1 : COARSE_SHIFT]
    actions = raw_actions[:ACTION_HORIZON]

    state_delta = np.where(DELTA_MASK[:ACTION_DIM], state[:ACTION_DIM], 0.0).astype(np.float32)
    coarse_actions = _normalize_action_chunk(coarse_actions, state_delta)
    actions = _normalize_action_chunk(actions, state_delta)

    return coarse_actions, actions


def main(args: Args) -> None:
    dataset_root = pathlib.Path(args.dataset_root)
    output_dir = pathlib.Path(args.output_dir)

    parquet_paths = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"No extracted parquet episodes found under {dataset_root}. Expected data/chunk-*/episode_*.parquet"
        )

    if args.max_episodes is not None:
        parquet_paths = parquet_paths[: args.max_episodes]

    info_path = dataset_root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        print(f"dataset_tasks={info.get('total_tasks')} dataset_episodes={info.get('total_episodes')} using={len(parquet_paths)}")

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
        "coarse_actions": normalize.RunningStats(),
    }

    total_frames = 0
    for parquet_path in tqdm.tqdm(parquet_paths, desc="Computing Go2 ACOT norm stats"):
        states, actions = _load_episode_arrays(parquet_path)
        total_frames += len(states)
        for idx in range(len(states)):
            state = _transform_state(states[idx])
            coarse_actions, fine_actions = _transform_actions(_build_action_window(actions, idx), state)
            stats["state"].update(state[None, :])
            stats["coarse_actions"].update(coarse_actions.reshape(-1, coarse_actions.shape[-1]))
            stats["actions"].update(fine_actions.reshape(-1, fine_actions.shape[-1]))

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize.save(output_dir, norm_stats)
    print(f"frames={total_frames}")
    print(f"wrote={output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
