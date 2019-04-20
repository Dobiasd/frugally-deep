#!/usr/bin/env python3
"""Save application models mentioned in Keras documentation
"""

import convert_model
import keras
from keras import backend as K

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
    print(f'Done converting {keras_file} to {fdeep_file}.')


def main():
    """Save famous example models in Keras-h5 and fdeep-json format."""
    assert K.backend() == "tensorflow"
    assert K.floatx() == "float32"
    assert K.image_data_format() == 'channels_last'

    save_model('densenet121', keras.applications.densenet.DenseNet121())
    save_model('densenet169', keras.applications.densenet.DenseNet169())
    save_model('densenet201', keras.applications.densenet.DenseNet201())
    # save_model('inceptionresnetv2', keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(299, 299, 3)))  # lambda
    save_model('inceptionv3', keras.applications.inception_v3.InceptionV3(input_shape=(299, 299, 3)))
    save_model('mobilenet', keras.applications.mobilenet.MobileNet())
    save_model('mobilenetv2', keras.applications.mobilenet_v2.MobileNetV2())
    save_model('nasnetlarge', keras.applications.nasnet.NASNetLarge(input_shape=(331, 331, 3)))
    save_model('nasnetmobile', keras.applications.nasnet.NASNetMobile(input_shape=(224, 224, 3)))
    save_model('resnet50', keras.applications.resnet50.ResNet50())
    save_model('vgg16', keras.applications.vgg16.VGG16())
    save_model('vgg19', keras.applications.vgg19.VGG19())
    save_model('xception', keras.applications.xception.Xception(input_shape=(299, 299, 3)))


if __name__ == "__main__":
    main()
