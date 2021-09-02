import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as tf
from torchvision import utils

from PIL import Image

import os
import sys

class MangroveDataset(Dataset):
    def __init__(self, folder_path, apply_transforms=False, transforms=None):
        """Args
            folder_path: path to the dataset. This should contain two folders `/images`, `/labels`
            apply_transforms: Are any transforms to be applied
            tranforms: Empty as of now
        """
        self.path = folder_path
        folders = os.listdir(self.path)
        assert "images" in folders, "images folder doesnot exist"
        assert "labels" in folders, "labels folder doesnot exist"

        if apply_transforms:
            self.transforms = transforms

            assert self.transforms is not None

        # images
        self.images = os.listdir(os.path.join(self.path, "/images"))
        self.labels = os.listdir(os.path.join(self.path, "/labels"))

        # PIL to Tensor Transform
        self.pil_to_tensor = tf.PILToTensor()

    def __len__(self):
        """Return the len of the dataset"""
        return len(os.listdir(os.path.join(self.path, "/images")))

    def __getitem__(self, idx):
        """Return an image and its label"""
        assert torch.is_tensor(idx)

        image = Image(self.images[idx])
        label = Image(self.labels[idx])

        # Convert PIL to tensor
        image = self.pil_to_tensor(image).to(torch.float)
        label = self.pil_to_tensor(label).to(torch.long)

        sample = {"image": image, "label": label}

        return sample