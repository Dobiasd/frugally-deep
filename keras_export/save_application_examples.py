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
    keras.applications.xception.Xception(input_shape=(299, 299, 3)).save(
        os.path.join(dir_path, file_name + ".h5"))


def save_vgg16(dir_path, file_name):
    """Save VGG16 model"""
    keras.applications.vgg16.VGG16().save(
        os.path.join(dir_path, file_name + ".h5"))


def save_vgg19(dir_path, file_name):
    """Save VGG19 model"""
    keras.applications.vgg19.VGG19().save(
        os.path.join(dir_path, file_name + ".h5"))


def save_resnet50(dir_path, file_name):
    """Save ResNet50 model"""
    keras.applications.resnet50.ResNet50().save(
        os.path.join(dir_path, file_name + ".h5"))


def save_inceptionv3(dir_path, file_name):
    """Save InceptionV3 model"""
    keras.applications.inception_v3.InceptionV3(input_shape=(299, 299, 3)).save(
        os.path.join(dir_path, file_name + ".h5"))


def save_inceptionvresnetv2(dir_path, file_name):
    """Save InceptionResNetV2 model"""
    keras.applications.inception_resnet_v2.InceptionResNetV2().save(
        os.path.join(dir_path, file_name + ".h5"))


def save_mobilenet(dir_path, file_name):
    """Save MobileNet model"""
    keras.applications.mobilenet.MobileNet().save(
        os.path.join(dir_path, file_name + ".h5"))


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
        # save_inceptionvresnetv2(dir_path, 'inceptionvresnetv2') # wait for pip
        # save_mobilenet(dir_path, 'mobilenet') #relu6

        # keras_export/export_model.py keras_export/inceptionv3.h5 keras_export/inceptionv3.json
        # keras_export/export_model.py keras_export/resnet50.h5 keras_export/resnet50.json
        # keras_export/export_model.py keras_export/vgg16.h5 keras_export/vgg16.json
        # keras_export/export_model.py keras_export/vgg19.h5 keras_export/vgg19.json
        # keras_export/export_model.py keras_export/xception.h5 keras_export/xception.json

        # keras_export/export_model.py keras_export/inceptionvresnetv2.h5 keras_export/inceptionvresnetv2.json
        # keras_export/export_model.py keras_export/mobilenet.h5 keras_export/mobilenet.json


if __name__ == "__main__":
    main()
