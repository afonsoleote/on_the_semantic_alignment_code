from torch import nn


class LinearModel:
    id = "nn_linear"

    @staticmethod
    def build(n_input_units):
        layers = list()

        layers.append(nn.Linear(n_input_units, 1))

        return nn.Sequential(*layers)

class StandardDenseModel:
    id = "nn_standard"

    @staticmethod
    def build(n_input_units):
        layers = list()

        # single linear neuron
        layers.append(nn.Linear(n_input_units, 10))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(10, 1))

        # return the constructed network architecture
        return nn.Sequential(*layers)

class MediumDenseModel:
    id = "nn_medium"

    @staticmethod
    def build(n_input_units):
        layers = list()

        layers.append(nn.Linear(n_input_units, 100))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(100, 10))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(10, 1))

        return nn.Sequential(*layers)

class LargeDenseModel:
    id = "nn_large"

    @staticmethod
    def build(n_input_units):
        layers = list()

        layers.append(nn.Linear(n_input_units, 256))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(256, 128))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(128, 32))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(32, 1))

        return nn.Sequential(*layers)

