#!/usr/bin/env python3
"""Save application models mentioned in Keras documentation
"""

import convert_model
import tensorflow

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
    save_model('densenet121', tensorflow.keras.applications.densenet.DenseNet121())
    save_model('densenet169', tensorflow.keras.applications.densenet.DenseNet169())
    save_model('densenet201', tensorflow.keras.applications.densenet.DenseNet201())
    # save_model('inceptionresnetv2', tensorflow.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(299, 299, 3)))  # lambda
    save_model('inceptionv3', tensorflow.keras.applications.inception_v3.InceptionV3(input_shape=(299, 299, 3)))
    save_model('mobilenet', tensorflow.keras.applications.mobilenet.MobileNet())
    save_model('mobilenetv2', tensorflow.keras.applications.mobilenet_v2.MobileNetV2())
    save_model('nasnetlarge', tensorflow.keras.applications.nasnet.NASNetLarge(input_shape=(331, 331, 3)))
    save_model('nasnetmobile', tensorflow.keras.applications.nasnet.NASNetMobile(input_shape=(224, 224, 3)))
    save_model('resnet101', tensorflow.keras.applications.ResNet101())
    save_model('resnet101v2', tensorflow.keras.applications.ResNet101V2())
    save_model('resnet152', tensorflow.keras.applications.ResNet152())
    save_model('resnet152v2', tensorflow.keras.applications.ResNet152V2())
    save_model('resnet50', tensorflow.keras.applications.ResNet50())
    save_model('resnet50v2', tensorflow.keras.applications.ResNet50V2())
    save_model('vgg16', tensorflow.keras.applications.vgg16.VGG16())
    save_model('vgg19', tensorflow.keras.applications.vgg19.VGG19())
    save_model('xception', tensorflow.keras.applications.xception.Xception(input_shape=(299, 299, 3)))


if __name__ == "__main__":
    main()
