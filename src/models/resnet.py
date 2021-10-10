import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, _resnet


class ResNet(nn.Module):
    """
    ResNet with projection head.
    """

    def __init__(self, base_model, out_dim, conv3x3=False, final_bn=False, final_dp=False, posterior_head=False, posterior_family='gaussian', eps=1e-6):
        super(ResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False),
                            "resnet50_pretrained": models.resnet50(pretrained=True),
                            "dp_resnet18": dpresnet18(pretrained=False)}
        self.final_bn = final_bn
        self.final_dp = final_dp
        self.posterior_head = posterior_head
        self.posterior_family = posterior_family
        if self.posterior_head:
            assert posterior_family in ['gaussian', 'gaussian_hib']

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        if conv3x3:
            resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            resnet.maxpool = nn.Identity()

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)

        if self.posterior_head:
            out_dim += 1

        self.l2 = nn.Linear(num_ftrs, out_dim, bias=False)

        if self.final_bn:
            self.bn = nn.BatchNorm1d(num_ftrs)

        self.eps = eps

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze(2).squeeze(2)

        x = self.l1(h)
        if self.final_bn:
            x = self.bn(x)
        x = F.relu(x)
        if self.final_dp:
            x = F.dropout(x, training=True)
        x = self.l2(x)

        if self.posterior_head:
            if self.posterior_family == 'gaussian': 
                loc, scale = x[:, :-1], x[:, -1].unsqueeze(1)  # circular co-variance!
                scale = scale 
            elif self.posterior_family == 'gaussian_hib':
                loc, scale = x[:, :-1], x[:, -1].unsqueeze(1)  # circular co-variance!
                scale = F.softplus(scale) # restrict stdev to be above 0
            else:
                raise Exception(f'Posterior {self.posterior_family} not supported.')
            return h, loc, scale

        return h, x


class SupResNet(nn.Module):

    def __init__(self, base_model, conv3x3=False, num_classes=10):
        super(SupResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(num_classes=num_classes, pretrained=False),
                            "resnet50": models.resnet50(num_classes=num_classes, pretrained=False),
                            "dp_resnet18": dpresnet18(pretrained=False)}
        resnet = self._get_basemodel(base_model)
        if conv3x3:
            resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            resnet.maxpool = nn.Identity()
        self.resnet = resnet

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        return None, self.resnet(x)


class BasicDropoutBlock(BasicBlock):

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # always have dropout!
        out = F.dropout2d(out, training=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def dpresnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicDropoutBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
