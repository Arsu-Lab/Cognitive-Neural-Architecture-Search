from torch import nn

LAYER_TYPES = ['conv', 'pool']
NON_LINEARITIES = {
    'Conv2D': nn.ReLU,
    'Linear': nn.ReLU,
}

# Define parameter ranges for layers
PARAM_RANGES = {
    'conv': {
        'out_channels': [64, 96, 128, 256, 384, 512],
        'kernel_size': [11, 9, 5, 3],
        'stride': [1, 2, 3, 4],
        'padding': [0, 1],
    },
    'pool': {
        'kernel_size': [2, 3],
        'stride': [2],
    },
    'fc': {
        'out_features': [64, 96, 128, 256, 384, 512, 1024],
    },
}