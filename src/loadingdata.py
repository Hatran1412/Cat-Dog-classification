import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from src.dataset import data_loader


# function to count number of parameters
def get_n_params(model):
    np = 0
    for p in list(model.parameters()):
        np += p.nelement()
    return np


input_size = 224*224 * 3  # images are 224*224 pixels and has 3 channels because of RGB color
output_size = 2  # there are 2 classes---Cats and Dogs

# number of subprocesses to use for data loading
num_workers = 0

# how many samples per batch to load
batch_size = 64

# define training and test data directories
data_dir = './data/'
train_dir = os.path.join(data_dir, 'training_set/')
test_dir = os.path.join(data_dir, 'test_set/')

# create transformers
image_size = (224, 224)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

test_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

# read data set using the custom class
train_dataset = data_loader(train_dir, transform=train_transform)
test_dataset = data_loader(test_dir, transform=test_transforms)

# load data using utils
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          num_workers=num_workers)

accuracy_list = []
