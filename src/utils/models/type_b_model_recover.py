from torch import nn

from src.utils.models.custom_layers import PermuteLayer


class ConvExp7Recover:
    @staticmethod
    def build(depth=3):
        layers = list()

        # l1 - conv1
        layers.append(nn.Conv2d(depth, 32, 3, padding='same'))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(32, eps=1e-3))

        # l2 - conv2
        layers.append(nn.Conv2d(32, 32, 3, padding='same'))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(32, eps=1e-3))

        # l3 - pool1
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout2d(p=0.2))
        # l4 - conv3
        layers.append(nn.Conv2d(32, 64, 3, padding='same'))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(64, eps=1e-3))

        # l5 - pool2
        layers.append(nn.MaxPool2d(2))
        # l6 - conv4
        layers.append(nn.Conv2d(64, 64, 3, padding='same'))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(64, eps=1e-3))

        # l7 - pool3
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout2d(p=0.2))
        # l8 - conv5
        layers.append(nn.Conv2d(64, 128, 3, padding='same'))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(128, eps=1e-3))

        # l9 - pool4
        layers.append(nn.MaxPool2d(2))
        # l10 - conv6
        layers.append(nn.Conv2d(128, 128, 3, padding='same'))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(128, eps=1e-3))

        # l11 - pool5
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout2d(p=0.2))
        # l12 - flatten
        layers.append(PermuteLayer(permutation=(0, 2, 3, 1)))
        layers.append(nn.Flatten())
        # l13 - fc1
        layers.append(nn.Linear(2048, 64))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(64, eps=1e-3))

        # l14 - fc2
        layers.append(nn.Linear(64, 32))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(32, eps=1e-3))

        # l15 - out
        layers.append(nn.Linear(32, 1))
        layers.append(nn.Sigmoid())

        # return the constructed network architecture
        return nn.Sequential(*layers)
