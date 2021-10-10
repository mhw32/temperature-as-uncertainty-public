"""
Loader for ImageNet. Borrowed heavily from
https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

import os
import torch.utils.data as data
from torchvision import datasets


class ImageNet(data.Dataset):

    def __init__(self, root, train=True, image_transforms=None, two_views=False):
        super().__init__()
        split_dir = 'train' if train else 'validation'
        imagenet_dir = os.path.join(root, split_dir)
        self.dataset = datasets.ImageFolder(imagenet_dir, image_transforms)
        self.two_views = two_views

    def __getitem__(self, index):
        if self.two_views:
            view1, label = self.dataset.__getitem__(index)
            view2, _ = self.dataset.__getitem__(index)
            return index, view1.float(), view2.float(), label
        else:
            view, label = self.dataset.__getitem__(index)
            return index, view.float(), label

    def __len__(self):
        return len(self.dataset)
