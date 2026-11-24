import torch.nn as nn

class AdultMLP:
    @staticmethod
    def build(in_dim):
        layers = list()
        # l1 - fc1
        layers.append(nn.Linear(in_dim, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.3))
        # l2 - fc2
        layers.append(nn.Linear(128, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.3))
        # l3 - out
        layers.append(nn.Linear(64, 1))
        return nn.Sequential(*layers)