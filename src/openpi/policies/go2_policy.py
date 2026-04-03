"""Policy transforms for the Go2 robot."""

import dataclasses
from typing import ClassVar
from collections.abc import Sequence
import numpy as np
import torch
import copy

import openpi.models.model as _model
import openpi.transforms as transforms


RGB_CAMERA_RENAME_MAP = {
    "top_head": "base_0_rgb",
    "hand_left": "left_wrist_0_rgb",
    "hand_right": "right_wrist_0_rgb",
}

DEPTH_CAMERA_RENAME_MAP = {
    "top_head": "base_0_depth",
    "hand_left": "left_wrist_0_depth",
    "hand_right": "right_wrist_0_depth",
}


def _to_numpy(array_like):
    if isinstance(array_like, torch.Tensor):
        return array_like.cpu().numpy()
    return np.asarray(array_like)


def _parse_rgb_image(image) -> np.ndarray:
    image = _to_numpy(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.ndim != 3:
        raise ValueError(f"RGB image must be 3D, got shape {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    elif image.shape[-1] != 3:
        raise ValueError(f"RGB image must have 3 channels, got shape {image.shape}")
    return image


def _parse_depth_image(depth_image) -> np.ndarray:
    depth_image = _to_numpy(depth_image)

    if depth_image.ndim == 2:
        depth_image = depth_image[..., None]
    elif depth_image.ndim == 3:
        if depth_image.shape[0] == 1:
            depth_image = np.moveaxis(depth_image, 0, -1)
        elif depth_image.shape[0] == 3:
            if not (np.allclose(depth_image[0], depth_image[1]) and np.allclose(depth_image[1], depth_image[2])):
                raise ValueError(f"Depth image channels differ, got shape {depth_image.shape}")
            depth_image = np.moveaxis(depth_image[:1], 0, -1)
        elif depth_image.shape[-1] == 3:
            if not (np.allclose(depth_image[..., 0], depth_image[..., 1]) and np.allclose(depth_image[..., 1], depth_image[..., 2])):
                raise ValueError(f"Depth image channels differ, got shape {depth_image.shape}")
            depth_image = depth_image[..., :1]
        elif depth_image.shape[-1] != 1:
            raise ValueError(f"Depth image must be single-channel, got shape {depth_image.shape}")
    else:
        raise ValueError(f"Depth image must be 2D or 3D, got shape {depth_image.shape}")

    if depth_image.shape[-1] != 1:
        raise ValueError(f"Depth image must end with a singleton channel, got shape {depth_image.shape}")

    if np.issubdtype(depth_image.dtype, np.floating):
        return depth_image.astype(np.float32, copy=False)
    return depth_image


def _parse_camera_group(data: dict, source_key: str, rename_map: dict[str, str], parser) -> dict[str, np.ndarray]:
    parsed = {}
    for camera, renamed_camera in rename_map.items():
        if camera not in data[source_key]:
            raise ValueError(f"Camera {camera} not found in {source_key}")
        parsed[renamed_camera] = parser(data[source_key][camera])
    return parsed


def _build_camera_mask(rename_map: dict[str, str]) -> dict[str, np.bool_]:
    return {renamed_camera: np.True_ for renamed_camera in rename_map.values()}


@dataclasses.dataclass(frozen=True)
class Go2Inputs(transforms.DataTransformFn):
    """Inputs for the Go2 policy.
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = tuple(RGB_CAMERA_RENAME_MAP)
    EXPECTED_DEPTH_CAMERAS: ClassVar[tuple[str, ...]] = tuple(DEPTH_CAMERA_RENAME_MAP)

    rename_map = RGB_CAMERA_RENAME_MAP
    depth_rename_map = DEPTH_CAMERA_RENAME_MAP

    def __call__(self, data: dict) -> dict:

        # Pad the proprioceptive input to the action dimension of the model
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        state = copy.deepcopy(state)
        # state[14:]  = state[14:] * 120
        if self.state_mask is not None:
            state[self.state_mask] = 0

        # Ensure state has correct shape [batch_size, state_dim]
        state = state.squeeze()

        images = _parse_camera_group(data, "images", self.rename_map, _parse_rgb_image)
        image_mask = _build_camera_mask(self.rename_map)


        # Prepare inputs dictionary
        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }

        if "depth_images" in data:
            depth_images = _parse_camera_group(data, "depth_images", self.depth_rename_map, _parse_depth_image)
            inputs["depth_image"] = depth_images
            inputs["depth_image_mask"] = _build_camera_mask(self.depth_rename_map)

        # Add actions if present
        if "actions" in data:
            actions = data["actions"]
            if self.action_mask is not None:
                actions[:, self.action_mask[:actions.shape[1]]] = 0
            actions = transforms.pad_to_dim(actions, self.action_dim)



            inputs["actions"] = actions.squeeze()

        # Add prompt if present
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Go2Outputs(transforms.DataTransformFn):
    """Outputs for the Go2 policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :22])} 


@dataclasses.dataclass(frozen=True)
class Go2ACOTInputs(transforms.DataTransformFn):
    """Inputs for the Go2 policy.
    """

    action_dim: int

    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    prompt_map_inject_to_training: dict[str, str] | None = None

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = tuple(RGB_CAMERA_RENAME_MAP)
    EXPECTED_DEPTH_CAMERAS: ClassVar[tuple[str, ...]] = tuple(DEPTH_CAMERA_RENAME_MAP)

    rename_map = RGB_CAMERA_RENAME_MAP
    depth_rename_map = DEPTH_CAMERA_RENAME_MAP
    acot_action_generation: Sequence[Sequence[int]] | None = None

    def slice_state_and_action(self, data):
        # Slice the state and action to the expected dimensions based on the original data shape
        state_indices = None
        if len(data["state"]) == 183:
            state_indices = list(range(54, 68)) + [0, 1] + list(range(99, 104))

        if len(data["state"]) == 159:
            state_indices = list(range(30, 44)) + [0, 1] + list(range(75, 80))
        if state_indices is not None:
            data["state"] = data["state"][state_indices]

        if "actions" in data:
            assert data["actions"].shape[1] == 40
            data["actions"] = np.column_stack((data["actions"][:, 16:30], data["actions"][:, 0:2], data["actions"][:, 33:38]))
        return data
    
    def random_inject_prompt(self, data):
        color_episode_pairs_for_task_sort_packages = {
            'white': [
                0, 9, 11, 15, 18, 19, 22, 34, 39, 41, 49, 52, 55, 62, 66, 68, 69, 73, 74, 81, 90, 96, 111, 120, 123, 125, \
                126, 129, 137, 139, 145, 149, 156, 157, 158, 164, 166, 170, 185, 187, 188, 190, 202, 207, 208, 209, 211, \
                213, 218, 220, 221, 226, 227, 228, 229, 230, 236, 246, 250, 251, 252, 260, 264, 266, 274, 275, 279, 282, 283
            ],
            'red': [
                1, 3, 5, 12, 13, 23, 24, 25, 26, 27, 28, 33, 38, 51, 53, 57, 58, 59, 61, 64, 76, 77, 80, 82, 84, 87, 91, 100, 102, 103, 105, \
                110, 114, 116, 118, 128, 142, 143, 148, 150, 152, 162, 163, 165, 167, 168, 173, 174, 176, 179, 186, 191, 192, 194, 197, 199, \
                205, 206, 214, 217, 222, 224, 234, 237, 238, 240, 241, 242, 243, 244, 247, 248, 253, 257, 262, 268, 270, 272, 273, 276, 277, 278, 284
            ],
            'black': [
                2, 14, 16, 17, 20, 29, 30, 31, 32, 37, 40, 42, 43, 44, 45, 46, 47, 48, 50, 56, 60, 65, 67, 70, 71, 72, 75, 78, 79, 83, 85, 88, \
                92, 95, 97, 99, 104, 107, 109, 112, 115, 117, 119, 122, 124, 127, 131, 132, 140, 141, 146, 147, 151, 155, 159, 160, 171, 172, 175, \
                177, 178, 180, 183, 184, 200, 203, 210, 212, 225, 233, 235, 245, 254, 255, 256, 259, 261, 263, 267, 269, 271, 280
            ],
            'yellow': [
                4, 6, 7, 8, 10, 21, 35, 36, 54, 63, 86, 89, 93, 94, 98, 101, 106, 108, 113, 121, 130, 133, 134, 135, 136, 138, 144, 153, \
                154, 161, 169, 181, 182, 189, 193, 195, 196, 198, 201, 204, 215, 216, 219, 223, 231, 232, 239, 249, 258, 265, 281, 285
            ]
        }

        task_name = data["task"]
        episode_idx = data["episode_index"]
        if self.prompt_map_inject_to_training is not None and task_name in self.prompt_map_inject_to_training:
            default_prompt = self.prompt_map_inject_to_training[task_name][0]
            inject_prob = self.prompt_map_inject_to_training[task_name][1]

            if task_name == "Sort packages":
                for key, value in color_episode_pairs_for_task_sort_packages.items():
                    if episode_idx in value:
                        default_prompt = default_prompt.replace("<color>", key)
                        break

            if np.random.rand() < inject_prob:
                data["prompt"] = default_prompt
    
        return data

    def __call__(self, data: dict) -> dict:
        data = self.slice_state_and_action(data)
        state = copy.deepcopy(transforms.pad_to_dim(data["state"], self.action_dim))
        if self.state_mask is not None:
            state[np.array(self.state_mask)] = 0

        images = _parse_camera_group(data, "images", self.rename_map, _parse_rgb_image)
        image_mask = _build_camera_mask(self.rename_map)

        # Prepare inputs dictionary
        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }

        if "depth_images" in data:
            depth_images = _parse_camera_group(data, "depth_images", self.depth_rename_map, _parse_depth_image)
            inputs["depth_image"] = depth_images
            inputs["depth_image_mask"] = _build_camera_mask(self.depth_rename_map)

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
                if self.action_mask is not None:
                    data[key][:, np.array(self.action_mask)[:data[key].shape[1]]] = 0
                data[key] = transforms.pad_to_dim(data[key], self.action_dim)
                inputs[key] = data[key]

        if "task" in data: # training
            data = self.random_inject_prompt(data)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Go2ACOTOutputs(transforms.DataTransformFn):
    """Outputs for the Go2 policy."""

    def __call__(self, data: dict) -> dict:
        keys = ['coarse_actions', 'actions']
        return {key: np.asarray(data[key][:, :21]) for key in keys if key in data}
