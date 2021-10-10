import torch.utils.data as data
from torchvision import datasets


class CelebA(data.Dataset):

    def __init__(self, root, train=True, image_transforms=None):
        super().__init__()
        self.dataset = datasets.ImageFolder(root, transform=image_transforms)

    def __getitem__(self, index):
        view, label = self.dataset.__getitem__(index)
        return index, view.float(), label

    def __len__(self):
        return len(self.dataset)

