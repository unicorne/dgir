"""
Data loading utilities for diffusion registration.

This module provides functions to load and preprocess medical imaging data
for registration tasks.
"""

import os
import torch
import numpy as np
import itk
from pathlib import Path
from typing import Tuple, List, Optional, Union
from ..training.config import Config


def load_tensor(path: Union[str, Path]) -> torch.Tensor:
    """
    Load tensor from file with proper error handling.
    
    Args:
        path: Path to tensor file.
        
    Returns:
        Loaded tensor.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        RuntimeError: If loading fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")
    
    try:
        return torch.load(path, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load tensor from {path}: {e}") from e


def load_preprocessed_data_2d(config: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load preprocessed 2D data for training.
    
    Args:
        config: Configuration object containing data paths.
        
    Returns:
        Tuple of (train_dxa, train_xray, test_dxa, test_xray) tensors.
        
    Raises:
        FileNotFoundError: If data files are not found.
        ValueError: If data shapes are inconsistent.
    """
    root = Path(config.data.data_root)
    
    # Load data files
    dxa_with_seg = load_tensor(root / 'affine_dxa_images_w_seg.pt')
    xray_with_seg = load_tensor(root / 'affine_radio_images_w_seg.pt')
    dxa_wo_seg = load_tensor(root / 'affine_dxa_images_wo_seg.pt')
    xray_wo_seg = load_tensor(root / 'affine_radio_images_wo_seg.pt')

    # Filter out problematic X-rays
    mask = np.ones(xray_wo_seg.shape[0], dtype=bool)
    mask[config.data.weird_xrays] = False

    train_xray = xray_wo_seg[mask, :]
    train_dxa = torch.vstack([dxa_wo_seg, dxa_with_seg[:75]])

    test_xray = xray_with_seg
    test_dxa = dxa_with_seg[75:]

    # Normalize if requested
    if config.data.normalize_images:
        train_xray = normalize_tensor(train_xray)
        test_xray = normalize_tensor(test_xray)

    return train_dxa, train_xray, test_dxa, test_xray


def load_preprocessed_data_3d(config: Config, max_subjects: Optional[int] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Load preprocessed 3D data for training.
    
    Args:
        config: Configuration object containing data paths.
        max_subjects: Maximum number of subjects to load. If None, uses config value.
        
    Returns:
        Tuple of (dataset_A, dataset_B) lists containing tensors.
        
    Raises:
        FileNotFoundError: If data files are not found.
    """
    if max_subjects is None:
        max_subjects = getattr(config.data, 'max_subjects', 300)
    
    dataset_A = []
    dataset_B = []
    count = 0
    
    for i in range(1, 458):
        if count >= max_subjects:
            break
            
        norm_path = f'/playpen-raid2/nurislam/diffreg/OASIS/OASIS_OAS1_{i:04}_MR1/aligned_norm.nii.gz'
        orig_path = f'/playpen-raid2/nurislam/diffreg/OASIS/OASIS_OAS1_{i:04}_MR1/aligned_orig.nii.gz'
        
        if not os.path.exists(norm_path) or not os.path.exists(orig_path):
            continue
            
        try:
            img_A = itk.imread(norm_path)
            img_B = itk.imread(orig_path)
            
            dataset_A.append(torch.from_numpy(np.asarray(img_A))[None])
            dataset_B.append(torch.from_numpy(np.asarray(img_B))[None])
            count += 1
            
        except Exception as e:
            print(f"Warning: Failed to load subject {i}: {e}")
            continue
    
    return dataset_A, dataset_B


def normalize_tensor(tensor: torch.Tensor, method: str = 'max') -> torch.Tensor:
    """
    Normalize tensor values.
    
    Args:
        tensor: Input tensor to normalize.
        method: Normalization method ('max', 'minmax', 'zscore').
        
    Returns:
        Normalized tensor.
    """
    if method == 'max':
        return tensor / tensor.amax(axis=list(range(1, tensor.ndim)), keepdim=True)
    elif method == 'minmax':
        min_val = tensor.amin(axis=list(range(1, tensor.ndim)), keepdim=True)
        max_val = tensor.amax(axis=list(range(1, tensor.ndim)), keepdim=True)
        return (tensor - min_val) / (max_val - min_val + 1e-8)
    elif method == 'zscore':
        mean_val = tensor.mean(axis=list(range(1, tensor.ndim)), keepdim=True)
        std_val = tensor.std(axis=list(range(1, tensor.ndim)), keepdim=True)
        return (tensor - mean_val) / (std_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


class RegistrationDataset:
    """Dataset class for registration tasks."""
    
    def __init__(self, images_A: List[torch.Tensor], images_B: List[torch.Tensor], 
                 device: str = 'cpu'):
        """
        Initialize dataset.
        
        Args:
            images_A: List of source images.
            images_B: List of target images.
            device: Device to move tensors to.
        """
        self.images_A = [img.to(device) for img in images_A] if isinstance(images_A, list) else images_A.to(device)
        self.images_B = [img.to(device) for img in images_B] if isinstance(images_B, list) else images_B.to(device)
        self.device = device
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.images_A)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        return self.images_A[idx], self.images_B[idx]
    
    def get_random_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get random batch of images.
        
        Args:
            batch_size: Size of batch to return.
            
        Returns:
            Tuple of (batch_A, batch_B).
        """
        import random
        
        if isinstance(self.images_A, list):
            indices = random.sample(range(len(self.images_A)), min(batch_size, len(self.images_A)))
            batch_A = torch.stack([self.images_A[i] for i in indices])
            batch_B = torch.stack([self.images_B[i] for i in indices])
        else:
            indices = random.sample(range(len(self.images_A)), min(batch_size, len(self.images_A)))
            batch_A = self.images_A[indices]
            batch_B = self.images_B[indices]
        
        return batch_A, batch_B

def load_custom_data(config: Config, with_masks=False, test_data = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load custom preprocessed 2D data for training.
    This function loads two sets of images and their corresponding masks.
    
    Args:
        config: Configuration object containing data paths.
        
    Returns:
        A tuple of tensors: (train_A, train_B, test_A, test_B)
    """
    root = Path(config.data.data_root)
    
    # --- Define your contrasts and file names ---
    # You can change these to load different contrasts
    contrast_a = config.data.contrast
    contrast_b = "DIXON"

    # --- Load Training Data ---
    train_a_path = root / f"train/{contrast_a}.pt"
    train_b_path = root / f"train/{contrast_b}.pt"
    train_a_masks_path = root / f"train/{contrast_a}_masks.pt"
    train_b_masks_path = root / f"train/{contrast_b}_masks.pt"
    
    print("Loading training data...")
    train_images_a = load_tensor(train_a_path)
    train_images_b = load_tensor(train_b_path)
    train_masks_a = load_tensor(train_a_masks_path)
    train_masks_b = load_tensor(train_b_masks_path)

    if test_data:
        split="test"
    else:
        split="val"

    train_a_path_val = root / f"{split}/{contrast_a}.pt"
    train_b_path_val = root / f"{split}/{contrast_b}.pt"
    train_a_masks_path_val = root / f"{split}/{contrast_a}_masks.pt"
    train_b_masks_path_val = root / f"{split}/{contrast_b}_masks.pt"

    print("Loading validation data to add to training set...")
    val_images_a = load_tensor(train_a_path_val)
    val_images_b = load_tensor(train_b_path_val)
    val_masks_a = load_tensor(train_a_masks_path_val)
    val_masks_b = load_tensor(train_b_masks_path_val)

    train_images_a = torch.cat((train_images_a, val_images_a), dim=0)
    train_images_b = torch.cat((train_images_b, val_images_b), dim=0)
    train_masks_a = torch.cat((train_masks_a, val_masks_a), dim=0)
    train_masks_b = torch.cat((train_masks_b, val_masks_b), dim=0)

    # Combine images with their masks as a second channel if needed
    # For now, we'll assume the model takes single-channel images and masks are handled separately.
    # If you need to stack them, you can do:
    # train_A = torch.cat((train_images_a, train_masks_a), dim=1)
    # train_B = torch.cat((train_images_b, train_masks_b), dim=1)
    train_A = train_images_a
    train_B = train_images_b
    train_A_masks = train_masks_a
    train_B_masks = train_masks_b

    # --- Load Validation/Test Data ---
    test_a_path = root / f"test/{contrast_a}.pt" # Using 'test' as test set
    test_b_path = root / f"test/{contrast_b}.pt"
    test_a_masks_path = root / f"test/{contrast_a}_masks.pt"
    test_b_masks_path = root / f"test/{contrast_b}_masks.pt"
    
    print("Loading validation data...")
    test_images_a = load_tensor(test_a_path)
    test_images_b = load_tensor(test_b_path)
    test_masks_a = load_tensor(test_a_masks_path)
    test_masks_b = load_tensor(test_b_masks_path)
    
    test_A = test_images_a
    test_B = test_images_b
    test_A_masks = test_masks_a
    test_B_masks = test_masks_b

    # Normalize if requested
    if config.data.normalize_images:
        print("Normalizing images...")
        train_A = normalize_tensor(train_A)
        train_B = normalize_tensor(train_B)
        test_A = normalize_tensor(test_A)
        test_B = normalize_tensor(test_B)


    print(f"Training set A shape: {train_A.shape}")
    print(f"Training set B shape: {train_B.shape}")
    print(f"Validation set A shape: {test_A.shape}")
    print(f"Validation set B shape: {test_B.shape}")
    if with_masks:
        return train_A, train_B, train_A_masks, train_B_masks, test_A, test_B, test_A_masks, test_B_masks
    else:
        return train_A, train_B, test_A, test_B


def create_data_loaders(config: Config, with_masks=False, test_data=False) -> Tuple[RegistrationDataset, RegistrationDataset]:
    """
    Create train and test data loaders based on configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    if config.model.dimension == 2:
        # Use our new custom data loading function
        if not with_masks:
            train_A, train_B, test_A, test_B = load_custom_data(config, with_masks=False, test_data=test_data)
            train_dataset = RegistrationDataset(train_A, train_B, config.training.device)
            test_dataset = RegistrationDataset(test_A, test_B, config.training.device)
            return train_dataset, test_dataset
        else:
            train_A, train_B, train_A_masks, train_B_masks, test_A, test_B, test_A_masks, test_B_masks = load_custom_data(config, with_masks=True, test_data=test_data)
            train_dataset = RegistrationDataset(train_A, train_B, config.training.device)
            test_dataset = RegistrationDataset(test_A, test_B, config.training.device)
            train_masks_dataset = RegistrationDataset(train_A_masks, train_B_masks, config.training.device)
            test_masks_dataset = RegistrationDataset(test_A_masks, test_B_masks, config.training.device)
            return train_dataset, test_dataset, train_masks_dataset, test_masks_dataset

    else:
        raise ValueError(f"Unsupported dimension: {config.model.dimension}")
        return None
