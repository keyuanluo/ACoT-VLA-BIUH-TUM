from collections.abc import Iterator, Sequence
import multiprocessing
import os
from pathlib import Path
import random
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class SafeDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: SupportsIndex):
        try:
            return self.dataset[index]
        except Exception as e:
            print(f"[Data Load Error] Skipping index {index} due to: {e}")
            return None
    
    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'dataset'")
        
        return getattr(self.dataset, name)


def _is_local_lerobot_repo(repo_id: str) -> bool:
    """Return True when a repo id actually points at a local LeRobot dataset root."""
    try:
        return Path(repo_id).expanduser().exists()
    except OSError:
        return False


def _required_repack_source_keys(repack_transforms: _transforms.Group) -> set[str]:
    source_keys: set[str] = set()
    for transform in repack_transforms.inputs:
        if isinstance(transform, _transforms.RepackTransform):
            for source_key in _transforms.flatten_dict(transform.structure).values():
                if isinstance(source_key, str):
                    source_keys.add(source_key)
    return source_keys


def _select_required_video_keys(
    dataset_meta: lerobot_dataset.LeRobotDatasetMetadata, data_config: _config.DataConfig
) -> list[str]:
    """Infer which LeRobot video keys are actually consumed by the repack transform."""
    required_source_keys = _required_repack_source_keys(data_config.repack_transforms)
    required_video_keys = [key for key in dataset_meta.video_keys if key in required_source_keys]
    return required_video_keys or list(dataset_meta.video_keys)


class LocalFilteredLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """A local-only LeRobot loader that can ignore unused video modalities.

    Some competition datasets are unpacked incrementally. For smoke runs we only need the
    RGB streams referenced by the repack transform, so we avoid treating missing unused
    depth videos as a fatal download/cache issue.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        root: str | Path,
        selected_video_keys: Sequence[str] | None = None,
        episodes: list[int] | None = None,
        image_transforms: typing.Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        del download_videos  # Local datasets should never fall back to remote downloads here.
        torch.utils.data.Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(root)
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else lerobot_dataset.CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else lerobot_dataset.get_safe_default_codec()
        self.delta_indices = None

        # Unused attributes kept for parity with LeRobotDataset.
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = lerobot_dataset.LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        if selected_video_keys is not None:
            selected_video_keys = tuple(selected_video_keys)
            original_video_keys = set(self.meta.video_keys)
            self.meta.info["features"] = {
                key: value
                for key, value in self.meta.info["features"].items()
                if key not in original_video_keys or key in selected_video_keys
            }

        self.stats = self.meta.stats
        if self.episodes is not None and self.meta._version >= lerobot_dataset.packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = lerobot_dataset.aggregate_stats(episodes_stats)

        missing_files = [str(self.root / fpath) for fpath in self.get_episodes_file_paths() if not (self.root / fpath).is_file()]
        if missing_files:
            preview = ", ".join(missing_files[:5])
            extra = "" if len(missing_files) <= 5 else f" ... (+{len(missing_files) - 5} more)"
            raise FileNotFoundError(f"Local LeRobot dataset is missing required files: {preview}{extra}")

        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = lerobot_dataset.get_episode_data_index(self.meta.episodes, self.episodes)

        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        lerobot_dataset.check_timestamps_sync(
            timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s
        )

        if self.delta_timestamps is not None:
            lerobot_dataset.check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = lerobot_dataset.get_delta_indices(self.delta_timestamps, self.fps)


def _resolve_split_indices(
    candidate_indices: Sequence[int], *, dataset_split: str, val_split_ratio: float, seed: int
) -> list[int]:
    if dataset_split not in {"train", "val"}:
        raise ValueError(f"Unsupported dataset split: {dataset_split}")
    if val_split_ratio <= 0:
        if dataset_split == "val":
            raise ValueError("Validation split requested, but val_split_ratio is not positive.")
        return list(candidate_indices)

    shuffled = np.asarray(candidate_indices, dtype=np.int64).copy()
    if shuffled.size < 2:
        raise ValueError("Validation split requires at least two candidate samples.")

    np.random.default_rng(seed).shuffle(shuffled)
    val_size = max(1, int(round(shuffled.size * val_split_ratio)))
    val_size = min(val_size, shuffled.size - 1)

    if dataset_split == "val":
        return np.sort(shuffled[:val_size]).tolist()
    return shuffled[val_size:].tolist()


class IndexSampler(torch.utils.data.Sampler[int]):
    """Sampler over a fixed set of dataset indices with optional reshuffling each pass."""

    def __init__(self, indices: Sequence[int], *, shuffle: bool, seed: int):
        self._indices = list(indices)
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

    def __iter__(self):
        indices = list(self._indices)
        if self._shuffle:
            rng = random.Random(self._seed + self._epoch)
            rng.shuffle(indices)
            self._epoch += 1
        return iter(indices)

    def __len__(self) -> int:
        return len(self._indices)


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        if not hasattr(self._dataset, "_datasets"):
            item = self._dataset[index]
            return self._transform(item)
        else:
            idx = index.__index__()
            for d in self._dataset._datasets:
                if idx < len(d):
                    item = d[idx]
                    return self._transform(item)
                idx -= len(d)
            raise IndexError("Index out of range")

    def __len__(self) -> int:
        if not hasattr(self._dataset, "_datasets"):
            length = len(self._dataset)
        else:
            length = 0
            for item in self._dataset._datasets:
                length += len(item)
        return length


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if model_config.model_type == _model.ModelType.ACOT_VLA_PI0 or model_config.model_type == _model.ModelType.ACOT_VLA_PI05:

        acot_action_horizons = jnp.array((model_config.coarse_action_horizon, model_config.action_horizon))
        joint_action_shifts = jnp.array((data_config.joint_action_shifts))
        action_chunk_size = max(acot_action_horizons * joint_action_shifts).item()

    else:
        action_chunk_size = model_config.action_horizon

    if isinstance(repo_id, list):
        # If repo_id is a list, create a dataset for each repo_id and concatenate them.
        dataset_metas = [
            lerobot_dataset.LeRobotDatasetMetadata(r) for r in repo_id
        ]
        dataset = lerobot_dataset.MultiLeRobotDataset(
            repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_chunk_size)]
                for dataset_meta in dataset_metas
                for key in data_config.action_sequence_keys
            },
        )
        if data_config.prompt_from_task:
            for n, d in enumerate(dataset._datasets):
                dataset._datasets[n] = TransformedDataset(
                    d, [_transforms.PromptFromLeRobotTask(dataset_metas[n].tasks)]
                )
        if data_config.prompt_from_hl_instruction:
            for n, d in enumerate(dataset._datasets):
                dataset._datasets[n] = TransformedDataset(
                    d,[_transforms.PromptFromHighlevelInstruction(dataset_metas[n].info['instruction_segments'])]
                )
        for i, d in enumerate(dataset._datasets):
            print(f"Dataset {i} has {len(d)} frames.")

    else:
        local_root = Path(repo_id).expanduser() if _is_local_lerobot_repo(repo_id) else None
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=local_root)
        dataset_kwargs = {
            "delta_timestamps": {
                key: [t / dataset_meta.fps for t in range(action_chunk_size)]
                for key in data_config.action_sequence_keys
            },
        }
        if local_root is not None:
            dataset = LocalFilteredLeRobotDataset(
                data_config.repo_id,
                root=local_root,
                selected_video_keys=_select_required_video_keys(dataset_meta, data_config),
                **dataset_kwargs,
            )
        else:
            dataset = lerobot_dataset.LeRobotDataset(
                data_config.repo_id,
                **dataset_kwargs,
            )

        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
        if data_config.prompt_from_hl_instruction:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromHighlevelInstruction(dataset_meta.info['instruction_segments'])])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    dataset_split: str = "train",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training."""
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        if config.val_split_ratio > 0:
            raise NotImplementedError("Validation split is only implemented for torch datasets.")
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        dataset_split=dataset_split,
        val_split_ratio=config.val_split_ratio,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    dataset_split: str = "train",
    val_split_ratio: float = 0.0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    local_batch_size = batch_size // jax.process_count()

    base_sampler = None
    candidate_indices = list(range(len(dataset)))
    if data_config.dataloader_sampler != '':
        from openpi.training.sampler import FrameSampler
        base_sampler = FrameSampler(dataset, data_config.dataloader_sampler)
        candidate_indices = list(base_sampler.valid_indices)
        shuffle = False

    sampler = None
    if val_split_ratio > 0:
        split_indices = _resolve_split_indices(
            candidate_indices,
            dataset_split=dataset_split,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )
        if len(split_indices) < local_batch_size:
            raise ValueError(
                f"{dataset_split} split size ({len(split_indices)}) is smaller than the local batch size ({local_batch_size})."
            )
        print(
            f"Using {dataset_split} split with {len(split_indices)} frames from {len(candidate_indices)} candidates "
            f"(val_split_ratio={val_split_ratio:.4f})."
        )
        sampler = IndexSampler(split_indices, shuffle=dataset_split == "train", seed=seed)
        shuffle = False
        if dataset_split == "val" and num_batches is None:
            num_batches = max(1, len(split_indices) // local_batch_size)
    elif base_sampler is not None:
        sampler = base_sampler

    dataset = SafeDataset(dataset)
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        sampler=sampler,
    )

    if model_config.model_type == _model.ModelType.ACOT_VLA_PI0 or model_config.model_type == _model.ModelType.ACOT_VLA_PI05:
        return DataLoaderACOTImpl(data_config, data_loader)
    else:
        return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        sampler = None,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            sampler=sampler,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    filter_items = [x for x in items if x is not None]
    if not filter_items:
        raise ValueError("All items in the batch were dropped during data loading.")
    # return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *filter_items)

    def debug_stack(*args):
        arrays = [np.asarray(x) for x in args]
        try:
            return np.stack(arrays, axis=0)
        except ValueError as e:
            shapes = [x.shape for x in arrays]
            unique_shapes = set(shapes)
            print(f"\n======== DEBUG ERROR ========")
            print(f"Stacking failed!")
            print(f"Found varying shapes: {unique_shapes}")
            print(f"First 5 shapes: {shapes[:5]}")
            print(f"Sample data (first item): {arrays[0]}")
            print(f"=============================\n")
            raise e

    return jax.tree.map(debug_stack, *filter_items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]

class DataLoaderACOTImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"], batch["coarse_actions"]
