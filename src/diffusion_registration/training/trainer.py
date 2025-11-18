"""
Training utilities for diffusion registration networks.

This module provides a comprehensive training framework with logging,
checkpointing, and evaluation capabilities.
"""

import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from .config import Config
from ..core.models import DiffusionRegistrationNet
from ..data.loaders import RegistrationDataset, create_data_loaders


class Trainer:
    """
    Main training class for diffusion registration networks.
    
    This class handles the complete training loop including:
    - Model training and validation
    - Checkpoint saving and loading
    - Loss logging and visualization
    - Learning rate scheduling
    """
    
    def __init__(self, net: DiffusionRegistrationNet, config: Config):
        """
        Initialize trainer.
        
        Args:
            net: Registration network to train.
            config: Training configuration.
        """
        self.net = net
        self.config = config
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), 
            lr=config.training.learning_rate
        )
        
        # Initialize tracking
        self.current_epoch = 0
        self.loss_history = {
            'train_all_loss': [],
            'train_similarity_loss': [],
            'train_regularization_loss': [],
            'val_all_loss': [],
            'val_similarity_loss': [],
            'val_regularization_loss': []
        }
        
        # Setup logging
        self._setup_logging()
        
        # Create output directories
        config.create_output_dirs()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = Path(self.config.output.log_dir) / 'training.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_dataset: RegistrationDataset) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_dataset: Training dataset.
            
        Returns:
            Dictionary of average losses for the epoch.
        """
        self.net.train()
        epoch_losses = {
            'all_loss': [],
            'similarity_loss': [],
            'regularization_loss': []
        }
        
        # Training loop
        batch_A, batch_B = train_dataset.get_random_batch(self.config.training.batch_size)
        
        self.optimizer.zero_grad()
        loss_object = self.net(batch_A, batch_B)
        
        # Backward pass
        loss_object.all_loss.backward()
        self.optimizer.step()
        
        # Record losses
        epoch_losses['all_loss'].append(loss_object.all_loss.item())
        epoch_losses['similarity_loss'].append(loss_object.similarity_loss.item())
        
        # Handle different loss types
        if hasattr(loss_object, 'bending_energy_loss'):
            epoch_losses['regularization_loss'].append(loss_object.bending_energy_loss.item())
        elif hasattr(loss_object, 'inverse_consistency_loss'):
            epoch_losses['regularization_loss'].append(loss_object.inverse_consistency_loss.item())
        else:
            epoch_losses['regularization_loss'].append(0.0)
        
        # Calculate averages
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def validate(self, val_dataset: RegistrationDataset) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_dataset: Validation dataset.
            
        Returns:
            Dictionary of validation losses.
        """
        self.net.eval()
        
        with torch.no_grad():
            batch_A, batch_B = val_dataset.get_random_batch(
                min(self.config.training.batch_size, len(val_dataset))
            )
            loss_object = self.net(batch_A, batch_B)
            
            val_losses = {
                'all_loss': loss_object.all_loss.item(),
                'similarity_loss': loss_object.similarity_loss.item(),
            }
            
            # Handle different loss types
            if hasattr(loss_object, 'bending_energy_loss'):
                val_losses['regularization_loss'] = loss_object.bending_energy_loss.item()
            elif hasattr(loss_object, 'inverse_consistency_loss'):
                val_losses['regularization_loss'] = loss_object.inverse_consistency_loss.item()
            else:
                val_losses['regularization_loss'] = 0.0
        
        return val_losses
    
    def train(self, train_dataset: RegistrationDataset, 
              val_dataset: Optional[RegistrationDataset] = None):
        """
        Main training loop.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Optional validation dataset.
        """
        self.logger.info(f"Starting training for {self.config.training.epochs} epochs")
        self.logger.info(f"Batch size: {self.config.training.batch_size}")
        self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
        
        for epoch in tqdm(range(self.current_epoch, self.config.training.epochs), 
                         desc="Training"):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch(train_dataset)
            self.loss_history['train_all_loss'].append((epoch, train_losses['all_loss']))
            self.loss_history['train_similarity_loss'].append((epoch, train_losses['similarity_loss']))
            self.loss_history['train_regularization_loss'].append((epoch, train_losses['regularization_loss']))
            
            # Update loss history
            #for key, value in train_losses.items():
            #    self.loss_history[key].append(value)
            
            # Validation
            val_losses = None
            if val_dataset is not None and epoch % (self.config.training.print_every) == 0:
                val_losses = self.validate(val_dataset)
                self.loss_history['val_all_loss'].append((epoch, val_losses['all_loss']))
                self.loss_history['val_similarity_loss'].append((epoch, val_losses['similarity_loss']))
                self.loss_history['val_regularization_loss'].append((epoch, val_losses['regularization_loss']))
            
            # Logging
            if epoch % self.config.training.print_every == 0:
                log_msg = (f"[{epoch}] Train - Total: {train_losses['all_loss']:.4f}, "
                          f"Sim: {train_losses['similarity_loss']:.4f}, "
                          f"Reg: {train_losses['regularization_loss']:.4f}")
                
                if val_losses is not None:
                    log_msg += (f" | Val - Total: {val_losses['all_loss']:.4f}, "
                               f"Sim: {val_losses['similarity_loss']:.4f}, "
                               f"Reg: {val_losses['regularization_loss']:.4f}")
                
                self.logger.info(log_msg)
            
            # Save checkpoints
            if epoch % self.config.training.save_every == 0 and epoch > 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")

            if epoch % self.config.training.plot_every == 0 and epoch > 0:
                plot_path = Path(self.config.output.results_dir) / f'losses_epoch_{epoch}.png'
                self.plot_losses(save_path=str(plot_path))
        
        self.logger.info("Training completed!")
        self.save_checkpoint("final_model.pth")
        final_plot_path = Path(self.config.output.results_dir) / 'losses_final.png'
        self.plot_losses(save_path=str(final_plot_path))
    
    def save_checkpoint(self, filename: str):
        """
        Save training checkpoint.
        
        Args:
            filename: Name of checkpoint file.
        """
        checkpoint_path = Path(self.config.output.checkpoint_dir) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else None,
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Checkpoint dictionary.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load states
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.loss_history = checkpoint.get('loss_history', self.loss_history)
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint

    def plot_losses(self, save_path: Optional[str] = None):
        """
        Plot training and validation losses.
        
        Args:
            save_path: Optional path to save the plot.
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Helper function to plot curves
            def plot_curve(ax, train_data, val_data, title):
                if train_data:
                    epochs, values = zip(*train_data)
                    ax.plot(epochs, values, label='Train', alpha=0.8)
                if val_data:
                    epochs, values = zip(*val_data)
                    ax.plot(epochs, values, label='Validation', linestyle='--', marker='o', markersize=3)
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True)
                if train_data or val_data:
                    ax.legend()

            # Plot Total Loss
            plot_curve(axes[0], 
                       self.loss_history['train_all_loss'], 
                       self.loss_history['val_all_loss'], 
                       'Total Loss')
            
            # Plot Similarity Loss
            plot_curve(axes[1], 
                       self.loss_history['train_similarity_loss'], 
                       self.loss_history['val_similarity_loss'], 
                       'Similarity Loss')
            
            # Plot Regularization Loss
            plot_curve(axes[2], 
                       self.loss_history['train_regularization_loss'], 
                       self.loss_history['val_regularization_loss'], 
                       'Regularization Loss')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Loss plot saved: {save_path}")
                plt.close(fig)  # Close the figure to free up memory
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib not available, skipping plot")


def train_model(config_path: str):
    """
    Convenience function to train a model from config file.
    
    Args:
        config_path: Path to configuration YAML file.
    """
    # Load configuration
    config = Config(config_path)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create data loaders
    train_dataset, test_dataset = create_data_loaders(config)
    
    # Create network
    net = DiffusionRegistrationNet(config)
    net = net.to(config.device)
    
    # Create trainer
    trainer = Trainer(net, config)
    
    # Train
    trainer.train(train_dataset, test_dataset)
