import torch.utils.data as data
from torchvision import datasets


class CIFAR10(data.Dataset):

    def __init__(self, root, train=True, image_transforms=None, two_views=False):
        super().__init__()
        self.dataset = datasets.cifar.CIFAR10(root, train=train, download=True, transform=image_transforms)
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

