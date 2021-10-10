import clip
import torch
import torch.nn as nn


class ClipWrapper(nn.Module):

    def __init__(self, device):
        super(ClipWrapper, self).__init__()
        device = 'cuda:{}'.format(device)
        model, preprocess = clip.load("ViT-B/32", device=device)
        self.encoder = model
        self.preprocess = preprocess

    def forward(self, image):
        encoding = self.encoder.encode_image(image)
        return encoding.float()

