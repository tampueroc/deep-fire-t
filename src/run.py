import os
import argparse
import torch
from torch.utils.data import DataLoader
from data.dataset import FireDataset

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="FireDataset DataLoader Script")
    parser.add_argument('--data_dir', type=str, default="data/deep_crown_dataset/organized_spreads",
                        help="Path to the dataset directory (default: %(default)s)")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for DataLoader (default: %(default)s)")
    parser.add_argument('--sequence_length', type=int, default=6,
                        help="Sequence length for the dataset (default: %(default)s)")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help="Dataset split to use (default: %(default)s)")
    parser.add_argument('--weather', action='store_true',
                        help="Enable weather data processing")
    parser.add_argument('--topological_features', action='store_true',
                        help="Enable topological features processing")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of workers for DataLoader (default: %(default)s)")
    parser.add_argument('--pin_memory', action='store_true',
                        help="Enable pin_memory for DataLoader (default: False)")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help="Device to run the script on (default: %(default)s)")

    args = parser.parse_args()

    # Change the current working directory to the home directory
    home_directory = os.path.expanduser("~")
    os.chdir(home_directory)

    # Dataset initialization
    train_dataset = FireDataset(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        transform=None,
        split=args.split,
        weather=args.weather,
        topological_features=args.topological_features
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Iterate through DataLoader
    for input_tensor, isochrone_mask in train_loader:
        print('Input tensor shape:', input_tensor.shape)
        print('Isochrone mask shape:', isochrone_mask.shape)
        break

if __name__ == "__main__":
    main()
