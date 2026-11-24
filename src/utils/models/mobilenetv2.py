from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Model:
    @staticmethod
    def build(n_output_units,
              width_mult=1.0, dense_units=1280,
              weights=MobileNet_V2_Weights.IMAGENET1K_V2,
              progress=True):
        model = mobilenet_v2(weights=weights,
                             progress=progress,
                             width_mult=width_mult, )
        # num_classes=n_output_units)

        # modify last layers to accommodate for change in dense layer units
        if dense_units != 1280:
            last_conv2d_layer = model.features[-1][0]
            model.features[-1][0] = nn.Conv2d(
                in_channels=last_conv2d_layer.in_channels,
                out_channels=dense_units,
                kernel_size=last_conv2d_layer.kernel_size,
                stride=last_conv2d_layer.stride,
                padding=last_conv2d_layer.padding,
                dilation=last_conv2d_layer.dilation,
                groups=last_conv2d_layer.groups,
                bias=last_conv2d_layer.bias,
                padding_mode=last_conv2d_layer.padding_mode,
                device=last_conv2d_layer.weight.device,
                dtype=last_conv2d_layer.weight.dtype)

            last_batchnorm2d_layer = model.features[-1][1]
            model.features[-1][1] = nn.BatchNorm2d(
                num_features=dense_units,
                eps=last_batchnorm2d_layer.eps,
                momentum=last_batchnorm2d_layer.momentum,
                affine=last_batchnorm2d_layer.affine,
                track_running_stats=last_batchnorm2d_layer.track_running_stats,
                device=last_batchnorm2d_layer.weight.device,
                dtype=last_batchnorm2d_layer.weight.dtype)

        # modify last dense layer
        model.classifier[1] = nn.Linear(dense_units, n_output_units)
        return model