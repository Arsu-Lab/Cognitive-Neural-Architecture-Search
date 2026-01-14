from evolution.architecture import Architecture
import torch
from torch import nn
from collections import OrderedDict
import math
from torchvision.models import alexnet, vgg16

class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores th"e current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x

class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output

def CORnet_S():
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([ 
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return model

def get_model_by_archname(name):
  layers = [
      # Layer 1: First conv layer
      {
          'type': 'conv',
          'out_channels': 384,
          'kernel_size': 7,
          'stride': 4,
          'padding': 0,
      },
      # Layer 2: First pool layer
      {
          'type': 'pool',
          'kernel_size': 3,
          'stride': 2,
      },
      # Layer 3: Second conv layer
      {
          'type': 'conv',
          'out_channels': 512,
          'kernel_size': 11,
          'stride': 1,
          'padding': 5,
      },
      # Layer 4: Second pool layer
      {
          'type': 'pool',
          'kernel_size': 3,
          'stride': 2,
      },
      # Layer 5: Third conv layer
      {
          'type': 'conv',
          'out_channels': 512,
          'kernel_size': 3,
          'stride': 1,
          'padding': 1,
      },
      # Layer 6: Third pool layer
      {
          'type': 'pool',
          'kernel_size': 3,
          'stride': 2,
      },
      # Layer 7: Fourth conv layer
      {
          'type': 'conv',
          'out_channels': 512,
          'kernel_size': 9,
          'stride': 1,
          'padding': 4,
      },
      # Layer 8: Fifth conv layer
      {
          'type': 'conv',
          'out_channels': 512,
          'kernel_size': 5,
          'stride': 1,
          'padding': 2,
      },
      # Layer 9: Sixth conv layer
      {
          'type': 'conv',
          'out_channels': 512,
          'kernel_size': 5,
          'stride': 1,
          'padding': 2,
      }
  ]


  layers_v4 = [
      # Layer 1: First conv layer
      {
          'type': 'conv',
          'out_channels': 384,
          'kernel_size': 11,
          'stride': 3,
          'padding': 0,
      },
      # Layer 2: First pool layer
      {
          'type': 'pool',
          'kernel_size': 2,
          'stride': 2,
      },
      # Layer 3: Second conv layer
      {
          'type': 'conv',
          'out_channels': 512,
          'kernel_size': 9,
          'stride': 1,
          'padding': 4,
      },
      # Layer 4: Second pool layer
      {
          'type': 'pool',
          'kernel_size': 3,
          'stride': 2,
      },
      # Layer 5: Third conv layer
      {
          'type': 'conv',
          'out_channels': 512,
          'kernel_size': 5,
          'stride': 1,
          'padding': 2,
      },
      # Layer 6: Third pool layer
      {
          'type': 'pool',
          'kernel_size': 3,
          'stride': 2,
      }
  ]

  layers_v2 = [
      # Layer 1: First conv layer
      {
          'type': 'conv',
          'out_channels': 256,
          'kernel_size': 11,
          'stride': 3,
          'padding': 0,
      },
      # Layer 2: First pool layer
      {
          'type': 'pool',
          'kernel_size': 3,
          'stride': 2,
      },
      # Layer 3: Second conv layer
      {
          'type': 'conv',
          'out_channels': 256,
          'kernel_size': 3,
          'stride': 1,
          'padding': 1,
      },
      # Layer 4: Second pool layer
      {
          'type': 'pool',
          'kernel_size': 3,
          'stride': 2,
      },
      # Layer 5: Third conv layer
      {
          'type': 'conv',
          'out_channels': 384,
          'kernel_size': 5,
          'stride': 1,
          'padding': 2,
      },
      # Layer 6: Third pool layer
      {
          'type': 'pool',
          'kernel_size': 3,
          'stride': 2,
      }
  ]

  layers_V1 = [
      # Layer 1: First conv layer
      {
          'type': 'conv',
          'out_channels': 512,
          'kernel_size': 9,
          'stride': 2,
          'padding': 0,
      },
      # Layer 2: First pool layer
      {
          'type': 'pool',
          'kernel_size': 3,
          'stride': 2,
      },
      # Layer 3: Second conv layer
      {
          'type': 'conv',
          'out_channels': 512,
          'kernel_size': 5,
          'stride': 1,
          'padding': 2,
      },
      # Layer 4: Second pool layer
      {
          'type': 'pool',
          'kernel_size': 2,
          'stride': 2,
      },
      # Layer 5: Third conv layer
      {
          'type': 'conv',
          'out_channels': 512,
          'kernel_size': 9,
          'stride': 1,
          'padding': 4,
      },
      # Layer 6: Third pool layer
      {
          'type': 'pool',
          'kernel_size': 3,
          'stride': 2,
      }
  ]


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if name == "V2":
      selected_layers = layers_v2
      architecture = Architecture(layers=selected_layers)
      model = architecture.build_model(device)
      return model, architecture.init_weights
  elif name == "V4":
      selected_layers = layers_v4
      architecture = Architecture(layers=selected_layers)
      model = architecture.build_model(device)
      return model, architecture.init_weights
  elif name == "IT":
      selected_layers = layers
      architecture = Architecture(layers=selected_layers)
      model = architecture.build_model(device)
      return model, architecture.init_weights
  elif name == "CorNetS":
      def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
      
      model = CORnet_S().to(device)      
      return model, init_weights
  elif name == "VGG16":
      def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

      model = vgg16(weights=None).to(device)
      return model, init_weights
  elif name == "AlexNet":
      def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
      model = alexnet(weights=None).to(device)
      return model, init_weights

  return None, None