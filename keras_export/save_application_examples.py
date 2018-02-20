#!/usr/bin/env python3
"""Save application models mentioned in Keras documentation
"""

import os
import sys

import keras
from keras import backend as K

# Application model weights are not all available via tensorflow.python.keras
# in TensorFlow versions < 1.6.0.
# https://stackoverflow.com/questions/48810937/keras-tensorflow-does-not-find-weights-file-imagenet
# https://github.com/tensorflow/tensorflow/issues/16683#issuecomment-363621409
# That's why here the following imports are not used.
# from tensorflow.python import keras
# from tensorflow.python.keras import backend as K

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"


def save_xception(dir_path, file_name):
    """Save Xception model"""
    keras.applications.xception.Xception(input_shape=(299, 299, 3)).save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_vgg16(dir_path, file_name):
    """Save VGG16 model"""
    keras.applications.vgg16.VGG16().save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_vgg19(dir_path, file_name):
    """Save VGG19 model"""
    keras.applications.vgg19.VGG19().save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_resnet50(dir_path, file_name):
    """Save ResNet50 model"""
    keras.applications.resnet50.ResNet50().save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_inceptionv3(dir_path, file_name):
    """Save InceptionV3 model"""
    keras.applications.inception_v3.InceptionV3(input_shape=(299, 299, 3)).save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_inceptionvresnetv2(dir_path, file_name):
    """Save InceptionResNetV2 model"""
    keras.applications.inception_resnet_v2.InceptionResNetV2(
        input_shape=(299, 299, 3)).save(
            os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_mobilenet(dir_path, file_name):
    """Save MobileNet model"""
    keras.applications.mobilenet.MobileNet().save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_densenet201(dir_path, file_name):
    """Save DenseNet201 model"""
    keras.applications.densenet.DenseNet201().save(
        os.path.join(dir_path, file_name + ".h5"), include_optimizer=False)


def save_nasnetlarge(dir_path, file_name):
    """Save NASNetLarge model"""
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
        keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        keras.applications.nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
        # save_inceptionvresnetv2(dir_path, 'inceptionvresnetv2') # lambda
        # save_mobilenet(dir_path, 'mobilenet') # relu6

        # ../keras_export/convert_model.py inceptionv3.h5 inceptionv3.json
        # ../keras_export/convert_model.py resnet50.h5 resnet50.json
        # ../keras_export/convert_model.py vgg16.h5 vgg16.json
        # ../keras_export/convert_model.py vgg19.h5 vgg19.json
        # ../keras_export/convert_model.py xception.h5 xception.json
        # ../keras_export/convert_model.py densenet201.h5 densenet201.json
        # ../keras_export/convert_model.py nasnetlarge.h5 nasnetlarge.json

        # ../keras_export/convert_model.py inceptionvresnetv2.h5 inceptionvresnetv2.json
        # ../keras_export/convert_model.py mobilenet.h5 mobilenet.json


if __name__ == "__main__":
    main()
