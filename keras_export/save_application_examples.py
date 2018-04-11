#!/usr/bin/env python3
"""Save application models mentioned in Keras documentation
"""

import os
import sys

import keras
from keras import backend as K


__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"


def save_xception(dir_path, file_name):
    """Save Xception model"""
    print('save_xception')
    keras.applications.xception.Xception(input_shape=(299, 299, 3)).save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_vgg16(dir_path, file_name):
    """Save VGG16 model"""
    print('save_vgg16')
    keras.applications.vgg16.VGG16().save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_vgg19(dir_path, file_name):
    """Save VGG19 model"""
    print('save_vgg19')
    keras.applications.vgg19.VGG19().save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_resnet50(dir_path, file_name):
    """Save ResNet50 model"""
    print('save_resnet50')
    keras.applications.resnet50.ResNet50().save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_inceptionv3(dir_path, file_name):
    """Save InceptionV3 model"""
    print('save_inceptionv3')
    keras.applications.inception_v3.InceptionV3(input_shape=(299, 299, 3)).save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_inceptionvresnetv2(dir_path, file_name):
    """Save InceptionResNetV2 model"""
    print('save_inceptionvresnetv2')
    keras.applications.inception_resnet_v2.InceptionResNetV2(
        input_shape=(299, 299, 3)).save(
            os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_mobilenet(dir_path, file_name):
    """Save MobileNet model"""
    print('save_mobilenet')
    keras.applications.mobilenet.MobileNet().save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_densenet201(dir_path, file_name):
    """Save DenseNet201 model"""
    print('save_densenet201')
    keras.applications.densenet.DenseNet201().save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_nasnetlarge(dir_path, file_name):
    """Save NASNetLarge model"""
    print('save_nasnetlarge')
    keras.applications.nasnet.NASNetLarge().save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def main():
    """Save famous example models in Keras format."""
    if len(sys.argv) != 2:
        print('usage: [output dir]')
        sys.exit(1)
    else:
        assert K.backend() == "tensorflow"
        assert K.floatx() == "float32"
        assert K.image_data_format() == 'channels_last'

        dir_path = sys.argv[1]

        save_xception(dir_path, 'xception')
        save_vgg16(dir_path, 'vgg16')
        save_vgg19(dir_path, 'vgg19')
        save_resnet50(dir_path, 'resnet50')
        save_inceptionv3(dir_path, 'inceptionv3')
        save_densenet201(dir_path, 'densenet201')
        save_nasnetlarge(dir_path, 'nasnetlarge')
        save_mobilenet(dir_path, 'mobilenet')

        # save_inceptionvresnetv2(dir_path, 'inceptionvresnetv2') # lambda

        # ../keras_export/convert_model.py inceptionv3.h5 inceptionv3.json
        # ../keras_export/convert_model.py resnet50.h5 resnet50.json
        # ../keras_export/convert_model.py vgg16.h5 vgg16.json
        # ../keras_export/convert_model.py vgg19.h5 vgg19.json
        # ../keras_export/convert_model.py xception.h5 xception.json
        # ../keras_export/convert_model.py densenet201.h5 densenet201.json
        # ../keras_export/convert_model.py nasnetlarge.h5 nasnetlarge.json
        # ../keras_export/convert_model.py mobilenet.h5 mobilenet.json

        # ../keras_export/convert_model.py inceptionvresnetv2.h5 inceptionvresnetv2.json


if __name__ == "__main__":
    main()
