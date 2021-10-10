import random
from PIL import ImageFilter
from torchvision import transforms

IMAGE_SHAPE = {
    'cifar10': 32,
    'cifar100': 32,
    'tinyimagenet': 64,
    'imagenet': 256,
    'lsun': 256,
    'celeba': 256,
    'mscoco': 256,
}

CROP_SHAPE = {
    'cifar10': 32,
    'cifar100': 32,
    'tinyimagenet': 64,
    'imagenet': 224,
    'lsun': 224,
    'celeba': 224,
    'mscoco': 224,
}


def get_transforms(dataset):
    image_size = IMAGE_SHAPE[dataset]
    crop_size = CROP_SHAPE[dataset]
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms


class GaussianBlur(object):

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
