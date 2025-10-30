import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

import argparse
import numpy as np
import torch
from pathlib import Path
from itertools import combinations

def preprocess_and_save(input_dir: Path, output_dir: Path, contrast_a: str, contrast_b: str):
    """
    Processes MRI data from NumPy files and saves it in a PyTorch tensor format
    suitable for the diffusion-guided registration training.

    Args:
        input_dir (Path): The root directory of the raw data, containing
                          'train', 'val', and 'test' subdirectories.
        output_dir (Path): The directory where the processed .pt files
                           will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = input_dir / split
        if not split_dir.exists():
            print(f"Directory not found for split '{split}', skipping.")
            continue

        print(f"Processing split: {split}")
        print(f"Pairing {contrast_a} with {contrast_b}")

        # Define file paths
        file_a = split_dir / f"{contrast_a}.npy"
        file_b = split_dir / f"{contrast_b}.npy"
        mask_file_a = split_dir / f"{contrast_a}_masks.npy"
        mask_file_b = split_dir / f"{contrast_b}_masks.npy"

        # Load images and masks
        images_a = np.load(file_a)
        images_b = np.load(file_b)
        masks_a = np.load(mask_file_a)
        masks_b = np.load(mask_file_b)

        # Add a channel dimension: (num_samples, 256, 256) -> (num_samples, 1, 256, 256)
        images_a = np.expand_dims(images_a, axis=1)
        images_b = np.expand_dims(images_b, axis=1)
        masks_a = np.expand_dims(masks_a, axis=1)
        masks_b = np.expand_dims(masks_b, axis=1)

        # Convert to PyTorch tensors
        tensors_a = torch.from_numpy(images_a).float()
        tensors_b = torch.from_numpy(images_b).float()
        masks_tensors_a = torch.from_numpy(masks_a).float()
        masks_tensors_b = torch.from_numpy(masks_b).float()

        # Save the processed tensors
        output_subdir = output_dir / split
        output_subdir.mkdir(parents=True, exist_ok=True)

        torch.save(tensors_a, output_subdir / f"{contrast_a}.pt")
        torch.save(tensors_b, output_subdir / f"{contrast_b}.pt")
        torch.save(masks_tensors_a, output_subdir / f"{contrast_a}_masks.pt")
        torch.save(masks_tensors_b, output_subdir / f"{contrast_b}_masks.pt")

        print(f"Saved processed data for {contrast_a} and {contrast_b}")

def main():
    input_dir = "/home/students/studweilc1/MU-Diff/data/my_data2"
    output_dir = "own_data/my_data3"
    contrast_a = "T1_mapping_fl2d"
    contrast_b = "DIXON"
    preprocess_and_save(Path(input_dir), Path(output_dir), contrast_a, contrast_b)
    contrast_a = "BOLD"
    contrast_b = "DIXON"
    preprocess_and_save(Path(input_dir), Path(output_dir), contrast_a, contrast_b)
    contrast_a = "Diffusion"
    contrast_b = "DIXON"
    preprocess_and_save(Path(input_dir), Path(output_dir), contrast_a, contrast_b)

if __name__ == "__main__":
    main()