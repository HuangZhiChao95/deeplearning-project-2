import torch.nn as nn
from vgg.DCFConv2d import DCFConv2d
import torch

class VGG16(nn.Module):

    def __init__(self, num_bases, bases='FB'):
        config = [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512,
                  512, 'P']

        super(VGG16, self).__init__()
        layer = []
        in_channel = 3
        k = 0
        for l in config:
            if l == 'P':
                layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                if type(bases) == str:
                    layer.append(DCFConv2d(in_channel, l, kernel_size=3, num_bases=num_bases, padding=1, bases=bases))
                else:
                    layer.append(DCFConv2d(in_channel, l, kernel_size=3, num_bases=num_bases, padding=1, bases=bases[k]))
                    k += 1
                layer += [nn.BatchNorm2d(l), nn.ReLU(inplace=True)]
                in_channel = l

        self.feature = nn.Sequential(*layer)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.shape[0], -1)
        logits = self.classifier(feature)
        preds = torch.argmax(logits, dim=1)
        return logits, preds
