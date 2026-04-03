import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import json
from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download, nnx_utils
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image

config = _config.get_config("acot_icra_simulation_challenge_reasoning_to_action")
checkpoint_dir = "./checkpoints/acot_icra_simulation_challenge_reasoning_to_action/exp_name/30000"

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)
ckpt_root = os.path.dirname(checkpoint_dir)
steps = int(os.path.basename(checkpoint_dir))


data_config = config.data.create(config.assets_dirs, config.model)
base_dataset = _data_loader.create_torch_dataset(data_config, config.model)
dataset = _data_loader.TransformedDataset(
    base_dataset,
    [
        *data_config.repack_transforms.inputs,
    ],
)

max_inferences = 20 # set to determine max infer times
chunk_size = 30

all_gt_actions = []
all_inferred_actions = []

for i in range(max_inferences):
    data = dataset[i * chunk_size]
    if i >= max_inferences:
        break

    gt_actions = copy.deepcopy(data['actions'])
    if gt_actions.shape[1] in (36, 40):
        gt_actions = np.column_stack((gt_actions[:, 16:30], gt_actions[:, 0], gt_actions[:, 1], gt_actions[:, 33:38]))

    inferred_result = policy.infer(data)
    inferred_actions = inferred_result['actions']

    gt_actions = gt_actions.squeeze()[:inferred_actions.shape[0], ...]
    all_gt_actions.append(gt_actions)
    all_inferred_actions.append(inferred_actions)

if all_gt_actions:
    gt_actions_continuous = np.concatenate(all_gt_actions, axis=0)
    inferred_actions_continuous = np.concatenate(all_inferred_actions, axis=0)
    total_steps, num_dims = inferred_actions_continuous.shape
    time_steps_per_inference = gt_actions_continuous.shape[0] // max_inferences

    fig, axes = plt.subplots(6, 4, figsize=(20, 28), sharex=True)
    axes = axes.flatten()
    
    x_axis = np.arange(total_steps)
    for dim_idx in range(num_dims):
        ax = axes[dim_idx]
        
        # Plot the continuous action sequences
        ax.plot(x_axis, gt_actions_continuous[:, dim_idx], label='Ground Truth', color='cornflowerblue', alpha=0.9)
        ax.plot(x_axis, inferred_actions_continuous[:, dim_idx], label='Inferred', color='tomato', linestyle='--', alpha=0.9)
        
        # Mark the starting point of each inference sequence
        start_indices = np.arange(0, total_steps, time_steps_per_inference)
        ax.scatter(start_indices, gt_actions_continuous[start_indices, dim_idx], c='blue', marker='o', s=40, zorder=5, label='GT Start')
        ax.scatter(start_indices, inferred_actions_continuous[start_indices, dim_idx], c='darkred', marker='x', s=40, zorder=5, label='Inferred Start')
        
        ax.set_title(f'Action Dimension {dim_idx}')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

    # Set common X-axis label
    fig.supxlabel(f'Continuous Timestep (across {max_inferences} inferences)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
    fig.suptitle(f'Comparison of Ground Truth and Inferred Actions @Step {steps}', fontsize=18)
    plt.savefig(f'{ckpt_root}/inferred_vs_gt_actions-{steps}-model.png', dpi=300, bbox_inches='tight')
    # plt.show()

else:
    print("No data was collected for plotting.")