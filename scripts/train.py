import csv
import dataclasses
import functools
import logging
import pathlib
import platform
from typing import Any
import os
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


class LocalMetricsLogger:
    """Persist reduced training metrics locally and keep an updated loss plot."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        *,
        resuming: bool,
        csv_name: str = "train_metrics.csv",
        plot_name: str = "loss.png",
        loss_key: str = "loss",
        plot_title: str = "Training Loss",
    ):
        self._output_dir = pathlib.Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._output_dir / csv_name
        self._plot_path = self._output_dir / plot_name
        self._loss_key = loss_key
        self._plot_title = plot_title
        self._fieldnames: list[str] | None = None
        self._history: list[dict[str, float]] = []

        if resuming and self._csv_path.exists():
            with self._csv_path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    self._fieldnames = list(reader.fieldnames)
                for row in reader:
                    parsed_row = {k: float(v) for k, v in row.items() if k and k != "step" and v}
                    if "step" in row and row["step"]:
                        parsed_row["step"] = int(float(row["step"]))
                    if parsed_row:
                        self._history.append(parsed_row)

    @property
    def csv_path(self) -> pathlib.Path:
        return self._csv_path

    @property
    def plot_path(self) -> pathlib.Path:
        return self._plot_path

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        row = {"step": int(step)}
        row.update({k: self._to_scalar(v) for k, v in metrics.items()})

        if self._fieldnames is None:
            self._fieldnames = ["step", *metrics.keys()]
        elif self._fieldnames != ["step", *metrics.keys()]:
            raise ValueError(
                f"Metric schema changed during training. Expected {self._fieldnames}, got {['step', *metrics.keys()]}"
            )

        write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        with self._csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in self._fieldnames})

        self._history.append(row)
        self._write_loss_plot()

    @staticmethod
    def _to_scalar(value: Any) -> float:
        array = np.asarray(value)
        return float(array.item() if array.shape == () else array.mean())

    def _write_loss_plot(self) -> None:
        if not self._history or self._loss_key not in self._history[-1]:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logging.info("matplotlib is not available; skipping local loss plot generation")
            return

        steps = [int(row["step"]) for row in self._history if self._loss_key in row]
        losses = [float(row[self._loss_key]) for row in self._history if self._loss_key in row]
        if not steps:
            return

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, losses, color="#2563eb", linewidth=2)
        ax.set_title(self._plot_title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self._plot_path, dpi=160)
        plt.close(fig)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info

@at.typecheck
def acot_train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, _model.CoarseActions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions,
        coarse_actions: _model.CoarseActions
    ):
        return model.compute_loss(rng, observation, actions, coarse_actions, train=True)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions, coarse_actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions, coarse_actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    del config
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    @at.typecheck
    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=False)
        return jnp.mean(chunked_loss)

    eval_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    loss = loss_fn(model, eval_rng, observation, actions)
    return {"val_loss": loss}


@at.typecheck
def acot_eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, _model.CoarseActions],
) -> dict[str, at.Array]:
    del config
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
    ):
        return model.compute_loss(rng, observation, actions, coarse_actions, train=False)

    eval_rng = jax.random.fold_in(rng, state.step)
    observation, actions, coarse_actions = batch
    loss = loss_fn(model, eval_rng, observation, actions, coarse_actions)
    return {"val_loss": loss}


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite if not os.getenv("DEBUG_MODE", default=False) == "true" else True,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    train_metrics = LocalMetricsLogger(config.checkpoint_dir / "metrics", resuming=resuming)
    logging.info(
        "Local training metrics will be written to %s and %s",
        train_metrics.csv_path,
        train_metrics.plot_path,
    )
    val_metrics = None
    if config.val_split_ratio > 0:
        val_metrics = LocalMetricsLogger(
            config.checkpoint_dir / "metrics",
            resuming=resuming,
            csv_name="val_metrics.csv",
            plot_name="val_loss.png",
            loss_key="val_loss",
            plot_title="Validation Loss",
        )
        logging.info(
            "Validation metrics will be written to %s and %s",
            val_metrics.csv_path,
            val_metrics.plot_path,
        )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        dataset_split="train",
    )
    val_data_loader = None
    if config.val_split_ratio > 0:
        val_data_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            num_batches=config.val_num_batches,
            dataset_split="val",
        )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    num_params = training_utils.count_parameters(train_state.params)
    logging.info(f"Total number of parameters: {num_params:,}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    if config.model.model_type == _model.ModelType.ACOT_VLA_PI05 or config.model.model_type == _model.ModelType.ACOT_VLA_PI0:
        ptrain_step = jax.jit(
            functools.partial(acot_train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    peval_step = None
    if val_data_loader is not None:
        if config.model.model_type == _model.ModelType.ACOT_VLA_PI05 or config.model.model_type == _model.ModelType.ACOT_VLA_PI0:
            peval_step = jax.jit(
                functools.partial(acot_eval_step, config),
                in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
                out_shardings=replicated_sharding,
            )
        else:
            peval_step = jax.jit(
                functools.partial(eval_step, config),
                in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
                out_shardings=replicated_sharding,
            )

    start_step = int(train_state.step)
    print("\n--- Trainable Parameters ---")
    model = nnx.merge(train_state.model_def, train_state.params)
    trainable_state = nnx.state(model, config.trainable_filter)
    logging.info(f"{training_utils.array_tree_to_info(trainable_state)}")
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    def flush_metrics(step: int, pending_infos: list[dict[str, at.Array]]) -> list[dict[str, at.Array]]:
        if not pending_infos:
            return []

        stacked_infos = common_utils.stack_forest(pending_infos)
        reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
        reduced_info = {k: float(np.asarray(v)) for k, v in reduced_info.items()}
        info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
        msg = f"Step {step}: {info_str}"
        pbar.write(msg)
        logging.info(msg)
        wandb.log(reduced_info, step=step)
        train_metrics.log(step, reduced_info)
        return []

    def run_validation(step: int) -> None:
        if val_data_loader is None or peval_step is None or val_metrics is None:
            return

        val_infos = []
        for val_batch_idx, val_batch in enumerate(val_data_loader):
            val_rng = jax.random.fold_in(train_rng, step * 1_000_003 + val_batch_idx)
            with sharding.set_mesh(mesh):
                val_info = peval_step(val_rng, train_state, val_batch)
            val_infos.append(val_info)

        if not val_infos:
            return

        stacked_infos = common_utils.stack_forest(val_infos)
        reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
        reduced_info = {k: float(np.asarray(v)) for k, v in reduced_info.items()}
        info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
        msg = f"Validation Step {step}: {info_str}"
        pbar.write(msg)
        logging.info(msg)
        wandb.log(reduced_info, step=step)
        val_metrics.log(step, reduced_info)

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            infos = flush_metrics(step, infos)
        if val_data_loader is not None and config.val_interval > 0 and step % config.val_interval == 0:
            run_validation(step)
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    infos = flush_metrics(config.num_train_steps - 1, infos)
    if val_data_loader is not None and config.val_interval > 0 and (config.num_train_steps - 1) % config.val_interval != 0:
        run_validation(config.num_train_steps - 1)
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
