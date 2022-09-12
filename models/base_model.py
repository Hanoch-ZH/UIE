import torch.nn as nn
import torch.nn.functional as f


class BaseModel(nn.Module):
    """
    simple FCN model, 14 CNN layers to generate an attention map
    """
    def __init__(self, ):
        super(BaseModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16, momentum=0.8),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.ReLU()
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU()
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU()
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU()
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32, momentum=0.8),
            nn.ReLU()
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16, momentum=0.8),
            nn.ReLU()
        )
        self.layer15 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        y = self.layer6(y)
        y = self.layer7(y)
        y = self.layer8(y)
        y = self.layer9(y)
        y = self.layer10(y)
        y = self.layer11(y)
        y = self.layer12(y)
        y = self.layer13(y)
        y = self.layer14(y)
        y = self.layer15(y)
        attention = f.sigmoid(y)
        output = attention * x
        return output
