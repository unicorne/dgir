import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

import argparse
import itk
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.diffusion_registration.training.config import Config
from src.diffusion_registration.core.models import DiffusionRegistrationNet
from src.diffusion_registration.core.wrappers import register_pair
from src.diffusion_registration.data.own_loaders import create_data_loaders
from src.diffusion_registration.core.wrappers import compute_warped_image_multiNC

import monai
from monai.metrics import DiceMetric
dice_metric = DiceMetric(include_background=True, reduction="mean")
LNCC_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=2)
MI_loss = monai.losses.GlobalMutualInformationLoss()
SSIM_loss = monai.losses.ssim_loss.SSIMLoss(spatial_dims=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(config, checkpoint_path, device):
    config.training.device = device
    net = DiffusionRegistrationNet(config)
    net.load_checkpoint(checkpoint_path)
    net.to(device)
    net.eval()
    return net

def main(checkpoint_path, config_path):
    config = Config(config_path)
    print("Name: ", config.output.name)
    net = load_model(config, checkpoint_path, device)
    train_dataset, test_dataset, train_masks_dataset, test_masks_dataset = create_data_loaders(config, with_masks=True)

    dce_befores = []
    dce_afters = []
    ssim_befores = []
    ssim_afters = []
    lncc_befores = []
    lncc_afters = []

    for index in range(len(test_dataset)):
        image_A_tensor, image_B_tensor = test_dataset[index]
        image_A_batch = image_A_tensor.unsqueeze(0).to(device)
        image_B_batch = image_B_tensor.unsqueeze(0).to(device)
        mask_A_tensor, mask_B_tensor = test_masks_dataset[index]
        mask_A_batch = mask_A_tensor.unsqueeze(0).to(device)
        mask_B_batch = mask_B_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
                # The forward pass of the network (e.g., BendingEnergyNet) calculates
                # the loss and also stores the results (phi_AB_vectorfield, warped_image_A)
                # as attributes of the 'net.net' object.
                #
                loss_object = net(image_A_batch, image_B_batch)

        transform_map = net.net.phi_AB_vectorfield
        spacing = net.net.spacing
        warped_A_batch = net.net.warped_image_A
        #print(f"Registration complete. Loss: {loss_object.all_loss.item():.4f}")

        with torch.no_grad():
                # Warp the mask using Nearest Neighbor (spline_order=0)
                #
                warped_mask_A_batch = compute_warped_image_multiNC(
                    mask_A_tensor.unsqueeze(0),  # Input must be float for grid_sample
                    transform_map,
                    spacing,
                    spline_order=0
                ).round()

        dce_before = dice_metric(
            y_pred=mask_B_batch.float(),
            y=mask_A_batch.float()
        ).item()

        dce_after = dice_metric(
            y_pred=warped_mask_A_batch.float(),
            y=mask_B_batch.float()
        ).item()
        dce_befores.append(dce_before)
        dce_afters.append(dce_after)

        ssim_before = SSIM_loss(image_A_batch, image_B_batch).item()
        ssim_after = SSIM_loss(warped_A_batch, image_B_batch).item()
        ssim_befores.append(ssim_before)
        ssim_afters.append(ssim_after)

        lncc_before = LNCC_loss(image_A_batch, image_B_batch).item()
        lncc_after = LNCC_loss(warped_A_batch, image_B_batch).item()
        lncc_befores.append(lncc_before)
        lncc_afters.append(lncc_after)

    mean_dce_before = np.mean(dce_befores)
    mean_dce_after = np.mean(dce_afters)
    mean_ssim_before = np.mean(ssim_befores)
    mean_ssim_after = np.mean(ssim_afters)
    mean_lncc_before = np.mean(lncc_befores)
    mean_lncc_after = np.mean(lncc_afters)

    print(f"Mean Dice Coefficient before registration: {mean_dce_before:.4f}")
    print(f"Mean Dice Coefficient after registration: {mean_dce_after:.4f}")
    print(f"Mean SSIM before registration: {mean_ssim_before:.4f}")
    print(f"Mean SSIM after registration: {mean_ssim_after:.4f}")
    print(f"Mean LNCC before registration: {mean_lncc_before:.4f}")
    print(f"Mean LNCC after registration: {mean_lncc_after:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained diffusion registration model.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration file."
    )
    args = parser.parse_args()
    main(args.checkpoint_path, args.config_path)


