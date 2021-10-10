import torch
import torch.nn as nn
from torchvision import models

class ByolWrapper(nn.Module):
    def __init__(self, device, model_path='/data/ozhang/resnet50_byol_imagenet2012.pth.tar'):
        super().__init__()
        base_model = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        checkpoint = torch.load(model_path)['online_backbone']
        state_dict = {}
        length = len(self.encoder.state_dict())
        for name, param in zip(self.encoder.state_dict(), list(checkpoint.values())[:length]):
            state_dict[name] = param
        self.encoder.load_state_dict(state_dict, strict=True)
        self.encoder.to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return x

