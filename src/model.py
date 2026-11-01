"""VGG6 for CIFAR-10 with a configurable activation function (architecture follows the course template)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# VGG-6 feature config from the template: 4 conv layers + 2 max-pools, followed by 1 linear classifier.
VGG6_CFG = [64, 64, "M", 128, 128, "M"]

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


def get_activation(name):
    """Return a fresh activation module by name."""
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


def make_layers(cfg, batch_norm=True, activation="relu"):
    """Build the conv feature extractor: each int is Conv3x3 (+BN) + activation, 'M' is 2x2 max-pool."""
    layers, in_channels = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            layers.append(get_activation(activation))
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, last_channels, num_classes=10):
        super().__init__()
        self.features = features
        self.classifier = nn.Linear(last_channels, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)  # global average pool -> (N, C, 1, 1)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _initialize_weights(self):
        """Same scheme as the template: He-normal convs, Kaiming linear, zero biases, BN starts as identity."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


def build_vgg6(activation="relu", batch_norm=True, num_classes=10):
    """Construct VGG6 with the chosen activation and optional batch norm."""
    last_channels = [v for v in VGG6_CFG if v != "M"][-1]
    return VGG(make_layers(VGG6_CFG, batch_norm, activation), last_channels, num_classes)


if __name__ == "__main__":
    # Sanity check: python -m src.model
    for act in ACTIVATIONS:
        model = build_vgg6(act)
        out = model(torch.randn(2, 3, 32, 32))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{act:8s} output {tuple(out.shape)}  params {n_params:,}")
