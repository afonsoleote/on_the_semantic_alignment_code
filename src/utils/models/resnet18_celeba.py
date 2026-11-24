from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18ModelCeleba:
    @staticmethod
    def build(weights=ResNet18_Weights.IMAGENET1K_V1, num_classes=1):
        model = resnet18(weights=weights)

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )

        return model