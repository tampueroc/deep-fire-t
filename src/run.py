import os
import torch
from torch.utils.data import DataLoader
from data.dataset import FireDataset

# Get the home directory
home_directory = os.path.expanduser("~")

# Change the current working directory to the home directory
os.chdir(home_directory)
data_dir = ""
transform = None
train_dataset = FireDataset(
    data_dir=data_dir,
    sequence_length=6,
    transform=transform,
    split='train',
    weather=False,
    topological_features=False
)

# Define batch size
batch_size = 8  # Adjust based on your computational resources

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

for (input_tensor, weather_tensor), isochrone_mask in train_loader:
    print('Input tensor shape:', input_tensor.shape)
    break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

