
import sys
path_to_pip_installs = "/tmp/test_env"
if path_to_pip_installs not in sys.path:
    sys.path.insert(0, path_to_pip_installs)

from pathlib import Path
import argparse

from src.diffusion_registration import Config, DiffusionRegistrationNet
from src.diffusion_registration.data.own_loaders import create_data_loaders
from src.diffusion_registration.training.trainer import train_model, Trainer

def main():
    parser = argparse.ArgumentParser(description="Train a diffusion registration model")
    parser.add_argument('--config', type=str, default="config/config_2d.yaml",
                        help='Path to the configuration file.')
    parser.add_argument('--data_root', type=str, default="own_data/my_data3",
                        help='Path to the root directory of the dataset.')
    args = parser.parse_args()
    # Load configuration
    config = Config(args.config)
    print("Configuration file parsed.")
    config.model.dimension = 2
    config.data.data_root = args.data_root
    print("Configuration loaded.")

    # Create data loaders
    train_dataset, test_dataset = create_data_loaders(config)
    print("Data loaders created.")

    # Initialize network
    net = DiffusionRegistrationNet(config)
    net = net.to(config.device)
    print("Network initialized.")

    trainer = Trainer(net, config)
    trainer.train(train_dataset, test_dataset)
    print("Training complete.")

    loss_plot_path = Path(config.output.results_dir) / 'training_losses.png'
    trainer.plot_losses(str(loss_plot_path))

if __name__ == "__main__":
    print("Starting training script...")
    main()