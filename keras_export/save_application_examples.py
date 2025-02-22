#!/usr/bin/env python3
"""Save application models mentioned in Keras documentation
"""

import keras
from keras.models import Model

import convert_model

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"


def save_model(file_name_base: str, model: Model) -> None:
    """Save and convert Keras model"""
    keras_file = f'{file_name_base}.keras'
    fdeep_file = f'{file_name_base}.json'
    print(f'Saving {keras_file}')
    model.save(keras_file)
    print(f'Converting {keras_file} to {fdeep_file}.')
    convert_model.convert(keras_file, fdeep_file)
    print(f'Conversion of model {keras_file} to {fdeep_file} done.')


def main() -> None:
    """Save famous example models in Keras-h5 and fdeep-json format."""
    print('Saving application examples')
    # save_model('convnextbase', keras.applications.convnext.ConvNeXtBase())  # custom object LayerScale
    # save_model('convnextlarge', keras.applications.convnext.ConvNeXtLarge())  # custom object LayerScale
    # save_model('convnextsmall', keras.applications.convnext.ConvNeXtSmall())  # custom object LayerScale
    # save_model('convnexttiny', keras.applications.convnext.ConvNeXtTiny())  # custom object LayerScale
    # save_model('convnextxlarge', keras.applications.convnext.ConvNeXtXLarge())  # custom object LayerScale
    save_model('densenet121', keras.applications.densenet.DenseNet121())
    save_model('densenet169', keras.applications.densenet.DenseNet169())
    save_model('densenet201', keras.applications.densenet.DenseNet201())
    save_model('efficientnetb0', keras.applications.efficientnet.EfficientNetB0(weights=None))
    save_model('efficientnetb1', keras.applications.efficientnet.EfficientNetB1(weights=None))
    save_model('efficientnetb2', keras.applications.efficientnet.EfficientNetB2(weights=None))
    save_model('efficientnetb3', keras.applications.efficientnet.EfficientNetB3(weights=None))
    save_model('efficientnetb4', keras.applications.efficientnet.EfficientNetB4(weights=None))
    save_model('efficientnetb5', keras.applications.efficientnet.EfficientNetB5(weights=None))
    save_model('efficientnetb6', keras.applications.efficientnet.EfficientNetB6(weights=None))
    save_model('efficientnetb7', keras.applications.efficientnet.EfficientNetB7(weights=None))
    save_model('efficientnetv2b0', keras.applications.efficientnet_v2.EfficientNetV2B0())
    save_model('efficientnetv2b1', keras.applications.efficientnet_v2.EfficientNetV2B1())
    save_model('efficientnetv2b2', keras.applications.efficientnet_v2.EfficientNetV2B2())
    save_model('efficientnetv2b3', keras.applications.efficientnet_v2.EfficientNetV2B3())
    save_model('efficientnetv2l', keras.applications.efficientnet_v2.EfficientNetV2L())
    save_model('efficientnetv2m', keras.applications.efficientnet_v2.EfficientNetV2M())
    save_model('efficientnetv2s', keras.applications.efficientnet_v2.EfficientNetV2S())
    # save_model('inceptionresnetv2', keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(299, 299, 3)))  # CustomScaleLayer
    save_model('inceptionv3', keras.applications.inception_v3.InceptionV3(input_shape=(299, 299, 3)))
    save_model('mobilenet', keras.applications.mobilenet.MobileNet())
    save_model('mobilenetv2', keras.applications.mobilenet_v2.MobileNetV2())
    save_model('nasnetlarge', keras.applications.nasnet.NASNetLarge(input_shape=(331, 331, 3)))
    save_model('nasnetmobile', keras.applications.nasnet.NASNetMobile(input_shape=(224, 224, 3)))
    save_model('resnet101', keras.applications.ResNet101())
    save_model('resnet101v2', keras.applications.ResNet101V2())
    save_model('resnet152', keras.applications.ResNet152())
    save_model('resnet152v2', keras.applications.ResNet152V2())
    save_model('resnet50', keras.applications.ResNet50())
    save_model('resnet50v2', keras.applications.ResNet50V2())
    save_model('vgg16', keras.applications.vgg16.VGG16())
    save_model('vgg19', keras.applications.vgg19.VGG19())
    save_model('xception', keras.applications.xception.Xception(input_shape=(299, 299, 3)))


if __name__ == "__main__":
    main()
