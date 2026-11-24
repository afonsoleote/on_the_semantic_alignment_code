from src.utils.models.FeatureExtractors import (ResNet_AvgPool_classifier, Bottleneck)

class ResNet50iNATModel:
    @staticmethod
    def build(block=Bottleneck, layers=None):
        if layers is None:
            layers = [3, 4, 6, 4]

        model = ResNet_AvgPool_classifier(block, layers)

        return model