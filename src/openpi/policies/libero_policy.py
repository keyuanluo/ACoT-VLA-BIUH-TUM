import dataclasses
import copy
import einops
import numpy as np
from collections.abc import Sequence
from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_depth_image(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    elif image.ndim == 3 and image.shape[0] in (1, 3):
        image = einops.rearrange(image, "c h w -> h w c")

    if image.ndim != 3:
        raise ValueError(f"Depth image must be HWC/CHW/2D, got shape {image.shape}")

    if image.shape[-1] > 1:
        image = image[..., :1]

    if np.issubdtype(image.dtype, np.integer):
        max_value = float(np.iinfo(image.dtype).max)
        image = image.astype(np.float32) / max(max_value, 1.0)
    else:
        image = image.astype(np.float32)

    return image


def _maybe_add_depth_inputs(data: dict, inputs: dict) -> dict:
    depth_inputs = {}
    depth_masks = {}

    if "observation/depth" in data:
        depth_inputs["base_0_depth"] = _parse_depth_image(data["observation/depth"])
        depth_masks["base_0_depth"] = np.True_

    if "observation/wrist_depth" in data:
        depth_inputs["left_wrist_0_depth"] = _parse_depth_image(data["observation/wrist_depth"])
        depth_masks["left_wrist_0_depth"] = np.True_

    if depth_inputs:
        inputs["depth_image"] = depth_inputs
        inputs["depth_image_mask"] = depth_masks

    return inputs


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return _maybe_add_depth_inputs(data, inputs)


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}

@dataclasses.dataclass(frozen=True)
class LiberoACOTInputs(transforms.DataTransformFn):

    model_type: _model.ModelType
    acot_action_generation: Sequence[Sequence[int]] | None = None
    def __call__(self, data: dict) -> dict:

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if self.acot_action_generation is not None and "actions" in data:
            action_horizons = self.acot_action_generation[0]
            joint_action_shifts = self.acot_action_generation[1]

            raw_data = data["actions"]
            keys = ["coarse_actions", "actions"]
            for idx, key in enumerate(keys):
                action_horizon = action_horizons[idx]
                joint_action_shift = joint_action_shifts[idx]
                required_length = (action_horizon - 1) * joint_action_shift + 1
                data[key] = copy.deepcopy(raw_data[:required_length:joint_action_shift])
                assert len(data[key]) == action_horizon
        
        for key in ['coarse_actions', 'actions']:
            if key in data:
                inputs[key] = data[key]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return _maybe_add_depth_inputs(data, inputs)


@dataclasses.dataclass(frozen=True)
class LiberoACOTOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        keys = ['coarse_actions', 'actions']
        ret_result = {key: np.asarray(data[key][:, :7]) for key in keys if key in data}
        return ret_result
