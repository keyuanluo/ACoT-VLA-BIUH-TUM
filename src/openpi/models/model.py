import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

ArrayT = TypeVar("ArrayT", at.Array, jax.ShapeDtypeStruct)

def convert_str_keys_to_int(d):
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            try:
                new_k = int(k) if isinstance(k, str) and k.isdigit() else k
            except ValueError:
                new_k = k
            new_d[new_k] = convert_str_keys_to_int(v)
        return new_d
    elif isinstance(d, (list, tuple)):
        return type(d)(convert_str_keys_to_int(x) for x in d)
    else:
        return d


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"
    ACOT_VLA_PI0 = "acot_pi0"
    ACOT_VLA_PI05 = "acot_pi05"

# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

DEPTH_IMAGE_KEYS = (
    "base_0_depth",
    "left_wrist_0_depth",
    "right_wrist_0_depth",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "depth_image": {
#         "base_0_depth": (float32|uint16)[*b, h, w, 1],  # Single-channel depth image
#         ...  # Additional depth camera views
#     },
#     "depth_image_mask": {
#         "base_0_depth": bool[*b],  # True if depth image is valid
#         ...  # Masks for additional depth views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#


def _normalize_rgb_image(image):
    """Convert uint8 RGB images to [-1, 1] float32 while leaving float inputs unchanged."""
    image = jnp.asarray(image)
    if image.dtype == jnp.uint8:
        return image.astype(jnp.float32) / 255.0 * 2.0 - 1.0
    return image


def _canonicalize_depth_image(depth_image):
    """Convert depth inputs to float32 channel-last single-channel images."""
    depth_image = jnp.asarray(depth_image)
    if depth_image.ndim == 2:
        depth_image = depth_image[..., None]
    elif depth_image.ndim == 3:
        depth_image = depth_image[..., None]
    elif depth_image.ndim >= 4:
        if depth_image.shape[-1] == 1:
            pass
        elif depth_image.shape[-3] == 1:
            depth_image = jnp.moveaxis(depth_image, -3, -1)
        else:
            raise ValueError(
                f"Depth image must be single-channel; expected trailing or leading singleton channel, got shape {depth_image.shape}"
            )
    else:
        raise ValueError(f"Depth image must have at least 2 dimensions, got shape {depth_image.shape}")

    if depth_image.shape[-1] != 1:
        raise ValueError(f"Depth image must be single-channel after canonicalization, got shape {depth_image.shape}")

    if depth_image.dtype != jnp.float32:
        depth_image = depth_image.astype(jnp.float32)
    return depth_image


@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]
    # Depth images, kept separate from RGB and stored as single-channel float32.
    depth_images: dict[str, at.Float[ArrayT, "*b hd wd dc"]] | None = None
    # Depth image masks, with same keys as depth_images.
    depth_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            data["image"][key] = _normalize_rgb_image(data["image"][key])
        depth_images = None
        if "depth_image" in data:
            depth_images = {}
            for key in data["depth_image"]:
                depth_images[key] = _canonicalize_depth_image(data["depth_image"][key])
        return cls(
            images=data["image"],
            image_masks={key: jnp.asarray(value) for key, value in data["image_mask"].items()},
            state=jnp.asarray(data["state"]),
            depth_images=depth_images,
            depth_image_masks=(
                {key: jnp.asarray(value) for key, value in data["depth_image_mask"].items()}
                if "depth_image_mask" in data
                else None
            ),
            tokenized_prompt=jnp.asarray(data["tokenized_prompt"]) if "tokenized_prompt" in data else None,
            tokenized_prompt_mask=(
                jnp.asarray(data["tokenized_prompt_mask"]) if "tokenized_prompt_mask" in data else None
            ),
            token_ar_mask=jnp.asarray(data["token_ar_mask"]) if "token_ar_mask" in data else None,
            token_loss_mask=jnp.asarray(data["token_loss_mask"]) if "token_loss_mask" in data else None,
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        if result["depth_images"] is not None:
            result["depth_image"] = result.pop("depth_images")
        else:
            result.pop("depth_images")
        if result["depth_image_masks"] is not None:
            result["depth_image_mask"] = result.pop("depth_image_masks")
        else:
            result.pop("depth_image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]
CoarseActions = at.Float[ArrayT, "*b ch ad"]

def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    depth_image_keys: Sequence[str] | None = None,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in default RGB/depth image masks (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    out_depth_images = None
    out_depth_masks = None
    if observation.depth_images is not None:
        if depth_image_keys is None:
            depth_image_keys = tuple(observation.depth_images)
        if not set(depth_image_keys).issubset(observation.depth_images):
            raise ValueError(
                f"depth_images dict missing keys: expected {depth_image_keys}, got {list(observation.depth_images)}"
            )

        out_depth_images = {}
        for key in depth_image_keys:
            depth_image = observation.depth_images[key]
            if depth_image.shape[-1] != 1:
                raise ValueError(f"Depth image {key} must remain single-channel, got shape {depth_image.shape}")
            out_depth_images[key] = depth_image

        out_depth_masks = {}
        for key in out_depth_images:
            if observation.depth_image_masks is None or key not in observation.depth_image_masks:
                out_depth_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
            else:
                out_depth_masks[key] = jnp.asarray(observation.depth_image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        depth_images=out_depth_images,
        depth_image_masks=out_depth_masks,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        params = convert_str_keys_to_int(params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Model params not found at: {params_path}")

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
