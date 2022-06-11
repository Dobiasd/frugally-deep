#!/usr/bin/env python3
"""Save application models mentioned in Keras documentation
"""

import convert_model
import tensorflow as tf

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"


def save_model(file_name_base, model):
    """Save and convert Keras model"""
    keras_file = f'{file_name_base}.h5'
    fdeep_file = f'{file_name_base}.json'
    print(f'Saving {keras_file}')
    model.save(keras_file, include_optimizer=False)
    print(f'Converting {keras_file} to {fdeep_file}.')
    convert_model.convert(keras_file, fdeep_file)
    print(f'Conversion of model {keras_file} to {fdeep_file} done.')


def main():
    """Save famous example models in Keras-h5 and fdeep-json format."""
    print('Saving application examples')
    save_model('densenet121', tf.keras.applications.densenet.DenseNet121())
    save_model('densenet169', tf.keras.applications.densenet.DenseNet169())
    save_model('densenet201', tf.keras.applications.densenet.DenseNet201())
    save_model('efficientnetb0', tf.keras.applications.efficientnet.EfficientNetB0())
    save_model('efficientnetb1', tf.keras.applications.efficientnet.EfficientNetB1())
    save_model('efficientnetb2', tf.keras.applications.efficientnet.EfficientNetB2())
    save_model('efficientnetb3', tf.keras.applications.efficientnet.EfficientNetB3())
    save_model('efficientnetb4', tf.keras.applications.efficientnet.EfficientNetB4())
    save_model('efficientnetb5', tf.keras.applications.efficientnet.EfficientNetB5())
    save_model('efficientnetb6', tf.keras.applications.efficientnet.EfficientNetB6())
    save_model('efficientnetb7', tf.keras.applications.efficientnet.EfficientNetB7())
    save_model('efficientnetv2b0', tf.keras.applications.efficientnet_v2.EfficientNetV2B0())
    save_model('efficientnetv2b1', tf.keras.applications.efficientnet_v2.EfficientNetV2B1())
    save_model('efficientnetv2b2', tf.keras.applications.efficientnet_v2.EfficientNetV2B2())
    save_model('efficientnetv2b3', tf.keras.applications.efficientnet_v2.EfficientNetV2B3())
    save_model('efficientnetv2l', tf.keras.applications.efficientnet_v2.EfficientNetV2L())
    save_model('efficientnetv2m', tf.keras.applications.efficientnet_v2.EfficientNetV2M())
    save_model('efficientnetv2s', tf.keras.applications.efficientnet_v2.EfficientNetV2S())
    # save_model('inceptionresnetv2', tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(299, 299, 3)))  # lambda
    save_model('inceptionv3', tf.keras.applications.inception_v3.InceptionV3(input_shape=(299, 299, 3)))
    save_model('mobilenet', tf.keras.applications.mobilenet.MobileNet())
    save_model('mobilenetv2', tf.keras.applications.mobilenet_v2.MobileNetV2())
    save_model('nasnetlarge', tf.keras.applications.nasnet.NASNetLarge(input_shape=(331, 331, 3)))
    save_model('nasnetmobile', tf.keras.applications.nasnet.NASNetMobile(input_shape=(224, 224, 3)))
    save_model('resnet101', tf.keras.applications.ResNet101())
    save_model('resnet101v2', tf.keras.applications.ResNet101V2())
    save_model('resnet152', tf.keras.applications.ResNet152())
    save_model('resnet152v2', tf.keras.applications.ResNet152V2())
    save_model('resnet50', tf.keras.applications.ResNet50())
    save_model('resnet50v2', tf.keras.applications.ResNet50V2())
    save_model('vgg16', tf.keras.applications.vgg16.VGG16())
    save_model('vgg19', tf.keras.applications.vgg19.VGG19())
    save_model('xception', tf.keras.applications.xception.Xception(input_shape=(299, 299, 3)))


if __name__ == "__main__":
    main()
