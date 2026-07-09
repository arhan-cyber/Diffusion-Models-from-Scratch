import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import PIL.Image

class EMNISTLettersDataset(Dataset):
    """
    A custom wrapper for the EMNIST letters dataset.
    - Swaps axes of PIL images (transposing) to correct EMNIST's default flipped orientation.
    - Maps labels from 1-26 (original) to 0-25 (standard 0-indexed classification).
    - Applies safe spatial augmentations (small rotations and translations) for training.
    """
    def __init__(self, data_dir="./data", train=True, augment=True):
        # Base transforms: transpose to correct orientation, convert to tensor, and normalize to [-1, 1]
        base_transforms = [
            transforms.Lambda(lambda img: img.transpose(PIL.Image.TRANSPOSE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
        
        if train and augment:
            # We insert the spatial augmentation BEFORE converting to tensor
            self.transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    fill=0
                ),
                *base_transforms
            ])
        else:
            self.transform = transforms.Compose(base_transforms)
            
        self.dataset = datasets.EMNIST(
            root=data_dir,
            split="letters",
            train=train,
            download=True,
            transform=self.transform
        )
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Shift EMNIST letters labels from [1, 26] to [0, 25]
        return img, label - 1
