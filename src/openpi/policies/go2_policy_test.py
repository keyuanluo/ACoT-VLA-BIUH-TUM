import numpy as np

from openpi.policies import go2_policy


def _make_go2_example():
    repeated_depth = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    return {
        "state": np.arange(8, dtype=np.float32),
        "images": {
            "top_head": np.ones((3, 6, 8), dtype=np.float32),
            "hand_left": np.ones((6, 8, 3), dtype=np.uint8),
            "hand_right": np.ones((3, 6, 8), dtype=np.float32),
        },
        "depth_images": {
            "top_head": np.arange(6 * 8, dtype=np.uint16).reshape(6, 8),
            "hand_left": np.arange(6 * 8, dtype=np.float32).reshape(1, 6, 8),
            "hand_right": np.stack([repeated_depth, repeated_depth, repeated_depth], axis=0),
        },
    }


def _assert_depth_branch(inputs: dict):
    assert set(inputs["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert set(inputs["depth_image"]) == {"base_0_depth", "left_wrist_0_depth", "right_wrist_0_depth"}

    assert inputs["image"]["base_0_rgb"].shape == (6, 8, 3)
    assert inputs["image"]["base_0_rgb"].dtype == np.uint8

    for key in ("base_0_depth", "left_wrist_0_depth", "right_wrist_0_depth"):
        assert inputs["depth_image"][key].shape == (6, 8, 1)

    assert "base_0_depth" not in inputs["image"]
    assert set(inputs["depth_image_mask"]) == {"base_0_depth", "left_wrist_0_depth", "right_wrist_0_depth"}


def test_go2_inputs_keep_depth_separate():
    transform = go2_policy.Go2Inputs(action_dim=32)
    inputs = transform(_make_go2_example())

    assert "depth_image" in inputs
    _assert_depth_branch(inputs)


def test_go2_acot_inputs_keep_depth_separate():
    transform = go2_policy.Go2ACOTInputs(action_dim=32)
    inputs = transform(_make_go2_example())

    assert "depth_image" in inputs
    _assert_depth_branch(inputs)


def test_parse_depth_image_rejects_mismatched_three_channel_depth():
    depth = np.zeros((3, 4, 5), dtype=np.float32)
    depth[1] = 1.0

    try:
        go2_policy._parse_depth_image(depth)
    except ValueError as exc:
        assert "channels differ" in str(exc)
    else:
        raise AssertionError("Expected mismatched three-channel depth input to raise ValueError")
