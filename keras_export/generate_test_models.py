#!/usr/bin/env python3
"""Generate a test model for frugally-deep.
"""

import sys
from typing import Tuple, List, Union

import numpy as np
from keras import activations
from keras.layers import ActivityRegularization
from keras.layers import AdditiveAttention
from keras.layers import Attention
from keras.layers import BatchNormalization, Concatenate, LayerNormalization, UnitNormalization
from keras.layers import CategoryEncoding, Embedding
from keras.layers import Conv1D, ZeroPadding1D, Cropping1D
from keras.layers import Conv2D, ZeroPadding2D, Cropping2D, CenterCrop
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D
from keras.layers import Identity, Conv2DTranspose, Conv1DTranspose
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import LeakyReLU, ELU, PReLU, ReLU, Softmax
from keras.layers import MaxPooling1D, AveragePooling1D, UpSampling1D
from keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.layers import MaxPooling3D, AveragePooling3D
from keras.layers import MultiHeadAttention
from keras.layers import Multiply, Add, Subtract, Average, Maximum, Minimum
from keras.layers import Normalization, Rescaling, Resizing
from keras.layers import Permute, Reshape, RepeatVector
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import ZeroPadding3D, Cropping3D
from keras.models import Model, load_model, Sequential

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"

NDFloat32Array = np.typing.NDArray[np.float32]
NDUInt32Array = np.typing.NDArray[np.int32]
Shape = Tuple[int, ...]
VariableShape = Tuple[Union[None, int], ...]


def replace_none_with(value: int, shape: VariableShape) -> Shape:
    """Replace every None with a fixed value."""
    return tuple(list(map(lambda x: x if x is not None else value, shape)))


def get_shape_for_random_data(data_size: int, shape: Shape) -> Shape:
    """Include size of data to generate into shape."""
    if len(shape) == 5:
        return (data_size, shape[0], shape[1], shape[2], shape[3], shape[4])
    if len(shape) == 4:
        return (data_size, shape[0], shape[1], shape[2], shape[3])
    if len(shape) == 3:
        return (data_size, shape[0], shape[1], shape[2])
    if len(shape) == 2:
        return (data_size, shape[0], shape[1])
    if len(shape) == 1:
        return (data_size, shape[0])
    raise ValueError('can not use shape for random data:', shape)


def generate_random_data(data_size: int, shape: VariableShape) -> NDFloat32Array:
    """Random data for training."""
    return np.random.random(
        size=get_shape_for_random_data(data_size, replace_none_with(42, shape))).astype(np.float32)


def generate_input_data(data_size: int, input_shapes: List[VariableShape]) -> List[NDFloat32Array]:
    """Random input data for training."""
    return [generate_random_data(data_size, input_shape)
            for input_shape in input_shapes]


def generate_integer_random_data(data_size: int, low: int, high: int, shape: Shape) -> NDUInt32Array:
    """Random data for training."""
    return np.random.randint(
        low=low, high=high, size=get_shape_for_random_data(data_size, replace_none_with(42, shape)))


def generate_integer_input_data(data_size: int, low: int, highs: List[int], input_shapes: List[Shape]) -> List[
    NDUInt32Array]:
    """Random input data for training."""
    return [generate_integer_random_data(data_size, low, high, input_shape)
            for high, input_shape in zip(highs, input_shapes)]


def as_list(value_or_values: Union[NDFloat32Array, List[NDFloat32Array]]) -> List[NDFloat32Array]:
    """Leave lists untouched, convert non-list types to a singleton list"""
    if isinstance(value_or_values, list):
        return value_or_values
    return [value_or_values]


def generate_output_data(data_size: int, outputs: List[NDFloat32Array]) -> List[NDFloat32Array]:
    """Random output data for training."""
    return [generate_random_data(data_size, output.shape[1:])
            for output in as_list(outputs)]


def get_test_model_exhaustive() -> Model:
    """Returns a exhaustive test model."""
    input_shapes: List[VariableShape] = [
        (2, 3, 4, 5, 6),  # 0
        (2, 3, 4, 5, 6),
        (7, 8, 9, 10),
        (7, 8, 9, 10),
        (11, 12, 13),
        (11, 12, 13),
        (14, 15),
        (14, 15),
        (16,),
        (16,),
        (2,),  # 10
        (1,),
        (2,),
        (1,),
        (1, 3),
        (1, 4),
        (1, 1, 3),
        (1, 1, 4),
        (1, 1, 1, 3),
        (1, 1, 1, 4),
        (1, 1, 1, 1, 3),  # 20
        (1, 1, 1, 1, 4),
        (26, 28, 3),
        (4, 4, 3),
        (4, 4, 3),
        (4,),
        (2, 3),
        (1,),
        (1,),
        (1,),
        (2, 3),  # 30
        (9, 16, 1),
        (1, 9, 16),
        (6, 1, 1),
        (1, 1, 1, 1, 6),
        (1, 1, 1, 10),
        (1, 1, 13),
        (1, 15),
        (1, 1, 1, 1, 6),
        (1, 1, 1, 5, 1),
        (1, 1, 4, 1, 1),  # 40
        (1, 3, 1, 1, 1),
        (2, 1, 1, 1, 1),
        (1, 1, 4, 1, 6),
        (1, 3, 1, 5, 1),
        (2, 1, 4, 1, 1),  # 45
        (1,),
        (3, 1),
        (6, 5, 4, 3, 2),
        (5, 4),
        (7, 4),  # 50
        (7, 4),
        (2, 4, 6, 8, 10),
    ]

    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []

    outputs.append(Conv1DTranspose(4, 3, padding='valid', use_bias=False)(inputs[6]))
    outputs.append(Conv2DTranspose(4, (5, 3), padding='valid', use_bias=False)(inputs[4]))

    outputs.append(Conv1DTranspose(4, 1, padding='valid', use_bias=False)(inputs[6]))
    outputs.append(Conv1DTranspose(4, 1, padding='same', use_bias=False)(inputs[6]))
    outputs.append(Conv1DTranspose(1, 3, padding='valid', use_bias=False)(inputs[6]))
    outputs.append(Conv1DTranspose(2, 3, padding='same')(inputs[6]))
    outputs.append(Conv1DTranspose(2, 5, padding='same', strides=2)(inputs[6]))
    outputs.append(Conv1DTranspose(2, 5, padding='valid', strides=2)(inputs[6]))
    # Current CPU implementations do not yet support dilation rates larger than 1
    # https://github.com/keras-team/keras/issues/20408
    # https://github.com/keras-team/keras/pull/20737
    #outputs.append(Conv1DTranspose(3, 5, padding='same', dilation_rate=2)(inputs[6]))
    #outputs.append(Conv1DTranspose(3, 5, padding='valid', dilation_rate=2)(inputs[6]))

    outputs.append(Conv2DTranspose(4, (1, 1), padding='valid', use_bias=False)(inputs[4]))
    outputs.append(Conv2DTranspose(4, (1, 1), padding='same', use_bias=False)(inputs[4]))
    outputs.append(Conv2DTranspose(1, (3, 3))(inputs[4]))
    outputs.append(Conv2DTranspose(4, (3, 3), padding='valid', use_bias=False)(inputs[4]))
    outputs.append(Conv2DTranspose(4, (3, 3), padding='same')(inputs[4]))
    outputs.append(Conv2DTranspose(4, (5, 5), padding='same', strides=(2, 3))(inputs[4]))
    outputs.append(Conv2DTranspose(4, (5, 5), padding='valid', strides=(2, 3))(inputs[4]))
    #outputs.append(Conv2DTranspose(4, (5, 5), padding='same', dilation_rate=(2, 3))(inputs[4]))
    #outputs.append(Conv2DTranspose(4, (5, 5), padding='valid', dilation_rate=(2, 3))(inputs[4]))

    outputs.append(Conv1DTranspose(1, 3, padding='valid')(inputs[6]))
    outputs.append(Conv1DTranspose(2, 1, padding='same')(inputs[6]))
    #outputs.append(Conv1DTranspose(3, 4, padding='same', dilation_rate=2)(inputs[6]))

    outputs.append(Conv2DTranspose(4, (3, 3))(inputs[4]))
    outputs.append(Conv2DTranspose(4, (3, 3), use_bias=False)(inputs[4]))
    outputs.append(Conv2DTranspose(4, (2, 4), strides=(2, 2), padding='same')(inputs[4]))
    #outputs.append(Conv2DTranspose(4, (2, 4), padding='same', dilation_rate=(2, 2))(inputs[4]))

    outputs.append(Conv1D(1, 3, padding='valid')(inputs[6]))
    outputs.append(Conv1D(2, 1, padding='same')(inputs[6]))
    outputs.append(Conv1D(2, 1, padding='same', strides=2)(inputs[6]))
    outputs.append(Conv1D(3, 4, padding='causal', dilation_rate=2)(inputs[6]))
    outputs.append(ZeroPadding1D(2)(inputs[6]))
    outputs.append(Cropping1D((2, 3))(inputs[6]))
    outputs.append(MaxPooling1D(2)(inputs[6]))
    outputs.append(MaxPooling1D(2, strides=2, padding='same')(inputs[6]))
    outputs.append(AveragePooling1D(2)(inputs[6]))
    outputs.append(AveragePooling1D(2, strides=2, padding='same')(inputs[6]))
    outputs.append(GlobalMaxPooling1D()(inputs[6]))
    outputs.append(GlobalAveragePooling1D()(inputs[6]))
    outputs.append(GlobalMaxPooling1D(keepdims=True)(inputs[6]))
    outputs.append(GlobalAveragePooling1D(keepdims=True)(inputs[6]))

    outputs.append(Normalization(axis=None, mean=2.1, variance=2.2)(inputs[4]))
    # outputs.append(Normalization(axis=-1, mean=2.1, variance=2.2)(inputs[6]))  # No longer supported in TensorFlow 2.16
    outputs.append(Normalization(axis=-1, mean=2.1, variance=2.2)(inputs[46]))
    outputs.append(Normalization(axis=1, mean=2.1, variance=2.2)(inputs[46]))
    outputs.append(Normalization(axis=-1, mean=2.1, variance=2.2)(inputs[47]))
    # outputs.append(Normalization(axis=1, mean=2.1, variance=2.2)(inputs[47]))  # No longer supported in TensorFlow 2.16
    outputs.append(Normalization(axis=2, mean=2.1, variance=2.2)(inputs[47]))
    for axis in range(1, 6):
        outputs.append(Normalization(axis=axis, mean=4.2, variance=2.3)(inputs[0]))

    outputs.append(Rescaling(23.5, 42.1)(inputs[0]))

    outputs.append(Conv2D(4, (3, 3))(inputs[4]))
    outputs.append(Conv2D(4, (3, 3), use_bias=False, padding='valid')(inputs[4]))
    outputs.append(Conv2D(4, (2, 4), strides=(2, 3), padding='same')(inputs[4]))
    outputs.append(Conv2D(4, (2, 4), padding='same', dilation_rate=(2, 3))(inputs[4]))

    outputs.append(SeparableConv2D(3, (3, 3))(inputs[4]))
    outputs.append(DepthwiseConv2D((3, 3))(inputs[4]))
    outputs.append(DepthwiseConv2D((1, 2))(inputs[4]))

    outputs.append(MaxPooling2D((2, 2))(inputs[4]))
    outputs.append(MaxPooling3D((2, 2, 2))(inputs[2]))
    outputs.append(MaxPooling2D((1, 3), strides=(2, 3), padding='same')(inputs[4]))
    outputs.append(MaxPooling3D((1, 3, 5), strides=(2, 3, 4), padding='same')(inputs[2]))
    outputs.append(AveragePooling2D((2, 2))(inputs[4]))
    outputs.append(AveragePooling3D((2, 2, 2))(inputs[2]))
    outputs.append(AveragePooling2D((1, 3), strides=(2, 3), padding='same')(inputs[4]))
    outputs.append(AveragePooling3D((1, 3, 5), strides=(2, 3, 4), padding='same')(inputs[2]))

    outputs.append(GlobalAveragePooling2D()(inputs[4]))
    outputs.append(GlobalAveragePooling3D()(inputs[2]))
    outputs.append(GlobalMaxPooling2D()(inputs[4]))
    outputs.append(GlobalMaxPooling3D()(inputs[2]))
    outputs.append(GlobalAveragePooling2D(keepdims=True)(inputs[4]))
    outputs.append(GlobalAveragePooling3D(keepdims=True)(inputs[2]))
    outputs.append(GlobalMaxPooling2D(keepdims=True)(inputs[4]))
    outputs.append(GlobalMaxPooling3D(keepdims=True)(inputs[2]))

    outputs.append(CenterCrop(4, 5)(inputs[4]))
    outputs.append(CenterCrop(5, 6)(inputs[4]))
    outputs.append(CenterCrop(19, 53)(inputs[23]))

    outputs.append(UpSampling2D(size=(1, 2), interpolation='nearest')(inputs[4]))
    outputs.append(UpSampling2D(size=(5, 3), interpolation='nearest')(inputs[4]))
    outputs.append(UpSampling2D(size=(1, 2), interpolation='bilinear')(inputs[4]))
    outputs.append(UpSampling2D(size=(5, 3), interpolation='bilinear')(inputs[4]))

    outputs.append(Resizing(4, 5)(inputs[4]))
    outputs.append(Resizing(5, 6)(inputs[4]))
    outputs.append(Resizing(19, 53, interpolation="bilinear")(inputs[23]))
    outputs.append(Resizing(19, 53, interpolation="nearest")(inputs[23]))
    # outputs.append(Resizing(7, 9, interpolation="area")(inputs[22]))  # No longer supported in TensorFlow 2.16
    # outputs.append(Resizing(19, 53, interpolation="area")(inputs[23]))  # No longer supported in TensorFlow 2.16
    outputs.append(Resizing(19, 53, crop_to_aspect_ratio=True)(inputs[23]))

    outputs.append(Permute((3, 4, 1, 5, 2))(inputs[0]))
    outputs.append(Permute((1, 5, 3, 2, 4))(inputs[0]))
    outputs.append(Permute((3, 4, 1, 2))(inputs[2]))
    outputs.append(Permute((2, 1, 3))(inputs[4]))
    outputs.append(Permute((2, 1))(inputs[6]))
    outputs.append(Permute((1,))(inputs[8]))
    outputs.append(Permute((3, 1, 2))(inputs[31]))
    outputs.append(Permute((3, 1, 2))(inputs[32]))

    outputs.append(BatchNormalization(center=False, scale=False)(inputs[11]))
    outputs.append(BatchNormalization()(inputs[11]))
    outputs.append(BatchNormalization()(inputs[10]))
    outputs.append(BatchNormalization()(inputs[14]))
    outputs.append(BatchNormalization()(inputs[26]))
    outputs.append(BatchNormalization()(inputs[23]))
    outputs.append(BatchNormalization()(inputs[0]))
    outputs.append(BatchNormalization(center=False)(inputs[0]))
    outputs.append(BatchNormalization(scale=False)(inputs[0]))
    outputs.append(BatchNormalization(center=False, scale=False)(inputs[0]))
    outputs.append(BatchNormalization()(inputs[0]))
    outputs.append(BatchNormalization(axis=1)(inputs[0]))
    outputs.append(BatchNormalization(axis=2)(inputs[0]))
    outputs.append(BatchNormalization(axis=3)(inputs[0]))
    outputs.append(BatchNormalization(axis=4)(inputs[0]))
    outputs.append(BatchNormalization(axis=5)(inputs[0]))
    outputs.append(BatchNormalization()(inputs[2]))
    outputs.append(BatchNormalization(axis=1)(inputs[2]))
    outputs.append(BatchNormalization(axis=2)(inputs[2]))
    outputs.append(BatchNormalization(axis=3)(inputs[2]))
    outputs.append(BatchNormalization(axis=4)(inputs[2]))
    outputs.append(BatchNormalization()(inputs[4]))
    # todo: check if TensorFlow >= 2.1 supports this
    # outputs.append(BatchNormalization(axis=1)(inputs[4])) # tensorflow.python.framework.errors_impl.InternalError:  The CPU implementation of FusedBatchNorm only supports NHWC tensor format for now.
    outputs.append(BatchNormalization(axis=2)(inputs[4]))
    outputs.append(BatchNormalization(axis=3)(inputs[4]))
    outputs.append(BatchNormalization()(inputs[6]))
    outputs.append(BatchNormalization(axis=1)(inputs[6]))
    outputs.append(BatchNormalization(axis=2)(inputs[6]))
    outputs.append(BatchNormalization()(inputs[8]))
    outputs.append(BatchNormalization(axis=1)(inputs[8]))
    outputs.append(BatchNormalization()(inputs[27]))
    outputs.append(BatchNormalization(axis=1)(inputs[27]))
    outputs.append(BatchNormalization()(inputs[14]))
    outputs.append(BatchNormalization(axis=1)(inputs[14]))
    outputs.append(BatchNormalization(axis=2)(inputs[14]))
    outputs.append(BatchNormalization()(inputs[16]))
    # todo: check if TensorFlow >= 2.1 supports this
    # outputs.append(BatchNormalization(axis=1)(inputs[16])) # tensorflow.python.framework.errors_impl.InternalError:  The CPU implementation of FusedBatchNorm only supports NHWC tensor format for now.
    outputs.append(BatchNormalization(axis=2)(inputs[16]))
    outputs.append(BatchNormalization(axis=3)(inputs[16]))
    outputs.append(BatchNormalization()(inputs[18]))
    outputs.append(BatchNormalization(axis=1)(inputs[18]))
    outputs.append(BatchNormalization(axis=2)(inputs[18]))
    outputs.append(BatchNormalization(axis=3)(inputs[18]))
    outputs.append(BatchNormalization(axis=4)(inputs[18]))
    outputs.append(BatchNormalization()(inputs[20]))
    outputs.append(BatchNormalization(axis=1)(inputs[20]))
    outputs.append(BatchNormalization(axis=2)(inputs[20]))
    outputs.append(BatchNormalization(axis=3)(inputs[20]))
    outputs.append(BatchNormalization(axis=4)(inputs[20]))
    outputs.append(BatchNormalization(axis=5)(inputs[20]))

    outputs.append(LayerNormalization()(inputs[11]))
    outputs.append(LayerNormalization()(inputs[10]))
    outputs.append(LayerNormalization()(inputs[26]))
    outputs.append(LayerNormalization()(inputs[24]))
    outputs.append(LayerNormalization()(inputs[0]))
    outputs.append(LayerNormalization(axis=1)(inputs[0]))
    outputs.append(LayerNormalization(axis=2)(inputs[0]))
    outputs.append(LayerNormalization(axis=3)(inputs[0]))
    outputs.append(LayerNormalization(axis=4)(inputs[0]))
    outputs.append(LayerNormalization(axis=5)(inputs[0]))
    outputs.append(LayerNormalization(axis=[1, 2])(inputs[0]))
    outputs.append(LayerNormalization(axis=[2, 3, 5])(inputs[0]))

    outputs.append(UnitNormalization()(inputs[11]))
    outputs.append(UnitNormalization()(inputs[10]))
    outputs.append(UnitNormalization()(inputs[26]))
    outputs.append(UnitNormalization()(inputs[24]))
    outputs.append(UnitNormalization()(inputs[0]))
    outputs.append(UnitNormalization(axis=1)(inputs[0]))
    outputs.append(UnitNormalization(axis=2)(inputs[0]))
    outputs.append(UnitNormalization(axis=3)(inputs[0]))
    outputs.append(UnitNormalization(axis=4)(inputs[0]))
    outputs.append(UnitNormalization(axis=5)(inputs[0]))
    outputs.append(UnitNormalization(axis=[1, 2])(inputs[0]))
    outputs.append(UnitNormalization(axis=[2, 3, 5])(inputs[0]))

    outputs.append(Dropout(0.5)(inputs[4]))
    outputs.append(ActivityRegularization(0.3, 0.4)(inputs[4]))

    outputs.append(ZeroPadding2D(2)(inputs[4]))
    outputs.append(ZeroPadding2D((2, 3))(inputs[4]))
    outputs.append(ZeroPadding2D(((1, 2), (3, 4)))(inputs[4]))
    outputs.append(Cropping2D(2)(inputs[4]))
    outputs.append(Cropping2D((2, 3))(inputs[4]))
    outputs.append(Cropping2D(((1, 2), (3, 4)))(inputs[4]))

    outputs.append(ZeroPadding3D(2)(inputs[2]))
    outputs.append(ZeroPadding3D((2, 3, 4))(inputs[2]))
    outputs.append(ZeroPadding3D(((1, 2), (3, 4), (5, 6)))(inputs[2]))
    outputs.append(Cropping3D(2)(inputs[2]))
    outputs.append(Cropping3D((2, 3, 4))(inputs[2]))
    outputs.append(Cropping3D(((1, 2), (3, 4), (2, 1)))(inputs[2]))

    outputs.append(Dense(3, use_bias=True)(inputs[0]))
    outputs.append(Dense(3, use_bias=True)(inputs[2]))
    outputs.append(Dense(3, use_bias=True)(inputs[4]))
    outputs.append(Dense(3, use_bias=True)(inputs[6]))
    outputs.append(Dense(3, use_bias=True)(inputs[8]))
    outputs.append(Dense(3, use_bias=True)(inputs[13]))
    outputs.append(Dense(3, use_bias=True)(inputs[14]))
    outputs.append(Dense(4, use_bias=False)(inputs[16]))
    outputs.append(Dense(4, use_bias=False, activation='tanh')(inputs[18]))
    outputs.append(Dense(4, use_bias=False)(inputs[20]))

    outputs.append(Reshape(((2 * 3 * 4 * 5 * 6),))(inputs[0]))
    outputs.append(Reshape((2, 3 * 4 * 5 * 6))(inputs[0]))
    outputs.append(Reshape((2, 3, 4 * 5 * 6))(inputs[0]))
    outputs.append(Reshape((2, 3, 4, 5 * 6))(inputs[0]))
    outputs.append(Reshape((2, 3, 4, 5, 6))(inputs[0]))

    outputs.append(Maximum()([inputs[0], inputs[1]]))
    outputs.append(Maximum()([inputs[2], inputs[3]]))
    outputs.append(Maximum()([inputs[4], inputs[5]]))
    outputs.append(Maximum()([inputs[6], inputs[7]]))
    outputs.append(Maximum()([inputs[8], inputs[9]]))

    outputs.append(Minimum()([inputs[0], inputs[1]]))
    outputs.append(Minimum()([inputs[2], inputs[3]]))
    outputs.append(Minimum()([inputs[4], inputs[5]]))
    outputs.append(Minimum()([inputs[6], inputs[7]]))
    outputs.append(Minimum()([inputs[8], inputs[9]]))

    # No longer works in TensorFlow 2.16, see: https://github.com/tensorflow/tensorflow/issues/65056
    # for normalize in [True, False]:
    # outputs.append(Dot(axes=(1, 1), normalize=normalize)([inputs[8], inputs[9]]))
    # outputs.append(Dot(axes=(1, 1), normalize=normalize)([inputs[0], inputs[10]]))
    # outputs.append(Dot(axes=1, normalize=normalize)([inputs[0], inputs[10]]))
    # outputs.append(Dot(axes=(3, 1), normalize=normalize)([inputs[31], inputs[32]]))
    # outputs.append(Dot(axes=(2, 3), normalize=normalize)([inputs[31], inputs[32]]))
    # outputs.append(Dot(axes=(2, 3), normalize=normalize)([inputs[14], inputs[16]]))
    # outputs.append(Dot(axes=(3, 2), normalize=normalize)([inputs[24], inputs[26]]))

    outputs.append(Reshape((16,))(inputs[8]))
    outputs.append(Reshape((2, 8))(inputs[8]))
    outputs.append(Reshape((2, 2, 4))(inputs[8]))
    outputs.append(Reshape((2, 2, 2, 2))(inputs[8]))
    outputs.append(Reshape((2, 2, 1, 2, 2))(inputs[8]))

    outputs.append(RepeatVector(3)(inputs[8]))

    outputs.append(ReLU()(inputs[0]))

    for axis in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
        outputs.append(Concatenate(axis=axis)([inputs[0], inputs[1]]))
    for axis in [-4, -3, -2, -1, 1, 2, 3, 4]:
        outputs.append(Concatenate(axis=axis)([inputs[2], inputs[3]]))
    for axis in [-3, -2, -1, 1, 2, 3]:
        outputs.append(Concatenate(axis=axis)([inputs[4], inputs[5]]))
    for axis in [-2, -1, 1, 2]:
        outputs.append(Concatenate(axis=axis)([inputs[6], inputs[7]]))
    for axis in [-1, 1]:
        outputs.append(Concatenate(axis=axis)([inputs[8], inputs[9]]))
    for axis in [-1]:  # [-1, 2] no longer supported in TensorFlow 2.16
        outputs.append(Concatenate(axis=axis)([inputs[14], inputs[15]]))
    for axis in [-1, 3]:
        outputs.append(Concatenate(axis=axis)([inputs[16], inputs[17]]))
    # for axis in [-1, 4]:
    # outputs.append(Concatenate(axis=axis)([inputs[18], inputs[19]]))  # no longer supported in TensorFlow 2.16
    # for axis in [-1, 5]:
    # outputs.append(Concatenate(axis=axis)([inputs[20], inputs[21]]))  # no longer supported in TensorFlow 2.16

    outputs.append(UpSampling1D(size=2)(inputs[6]))
    # outputs.append(UpSampling1D(size=2)(inputs[8])) # ValueError: Input 0 of layer up_sampling1d_1 is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: [None, 16]

    outputs.append(Multiply()([inputs[10], inputs[11]]))
    outputs.append(Multiply()([inputs[11], inputs[10]]))
    outputs.append(Multiply()([inputs[11], inputs[13]]))
    outputs.append(Multiply()([inputs[10], inputs[11], inputs[12]]))
    outputs.append(Multiply()([inputs[11], inputs[12], inputs[13]]))
    outputs.append(Multiply()([inputs[14], inputs[16], inputs[18], inputs[20]]))
    outputs.append(Multiply()([inputs[14], inputs[16]]))
    outputs.append(Multiply()([inputs[16], inputs[18]]))
    outputs.append(Multiply()([inputs[18], inputs[20]]))
    outputs.append(Multiply()([inputs[30], inputs[33]]))
    outputs.append(Multiply()([inputs[34], inputs[0]]))
    outputs.append(Multiply()([inputs[35], inputs[2]]))
    outputs.append(Multiply()([inputs[36], inputs[4]]))
    outputs.append(Multiply()([inputs[37], inputs[6]]))
    outputs.append(Multiply()([inputs[0], inputs[38]]))
    outputs.append(Multiply()([inputs[0], inputs[39]]))
    outputs.append(Multiply()([inputs[0], inputs[40]]))
    outputs.append(Multiply()([inputs[0], inputs[41]]))
    outputs.append(Multiply()([inputs[0], inputs[42]]))
    outputs.append(Multiply()([inputs[43], inputs[44]]))
    outputs.append(Multiply()([inputs[44], inputs[45]]))

    outputs.append(Attention(use_scale=False, score_mode='dot')([inputs[49], inputs[50]]))
    outputs.append(Attention(use_scale=False, score_mode='dot')([inputs[49], inputs[50], inputs[51]]))
    outputs.append(Attention(use_scale=True, score_mode='dot')([inputs[49], inputs[50]]))
    outputs.append(Attention(use_scale=False, score_mode='concat')([inputs[49], inputs[50]]))

    outputs.append(AdditiveAttention(use_scale=False)([inputs[49], inputs[50]]))
    outputs.append(AdditiveAttention(use_scale=False)([inputs[49], inputs[50], inputs[51]]))
    outputs.append(AdditiveAttention(use_scale=True)([inputs[49], inputs[50]]))
    outputs.append(AdditiveAttention(use_scale=True)([inputs[49], inputs[50], inputs[51]]))

    outputs.append(MultiHeadAttention(
        num_heads=1, key_dim=1, value_dim=None,
        use_bias=False, output_shape=None, attention_axes=None)(inputs[49], inputs[50]))
    outputs.append(MultiHeadAttention(
        num_heads=1, key_dim=1, value_dim=None,
        use_bias=True, output_shape=None, attention_axes=None)(inputs[49], inputs[50]))
    outputs.append(MultiHeadAttention(
        num_heads=1, key_dim=2, value_dim=None,
        use_bias=False, output_shape=None, attention_axes=None)(inputs[49], inputs[50]))
    outputs.append(MultiHeadAttention(
        num_heads=1, key_dim=2, value_dim=None,
        use_bias=True, output_shape=None, attention_axes=None)(inputs[49], inputs[50]))
    outputs.append(MultiHeadAttention(
        num_heads=1, key_dim=1, value_dim=2,
        use_bias=False, output_shape=None, attention_axes=None)(inputs[49], inputs[50]))
    outputs.append(MultiHeadAttention(
        num_heads=1, key_dim=1, value_dim=2,
        use_bias=True, output_shape=None, attention_axes=None)(inputs[49], inputs[50]))
    outputs.append(MultiHeadAttention(
        num_heads=3, key_dim=1, value_dim=None,
        use_bias=False, output_shape=None, attention_axes=None)(inputs[49], inputs[50]))
    outputs.append(MultiHeadAttention(
        num_heads=3, key_dim=1, value_dim=None,
        use_bias=True, output_shape=None, attention_axes=None)(inputs[49], inputs[50]))
    outputs.append(MultiHeadAttention(
        num_heads=1, key_dim=1, value_dim=None,
        use_bias=False, output_shape=None, attention_axes=None)(inputs[49], inputs[50], inputs[51]))
    outputs.append(MultiHeadAttention(
        num_heads=2, key_dim=3, value_dim=5,
        use_bias=False, output_shape=None, attention_axes=None)(inputs[49], inputs[50], inputs[51]))
    outputs.append(MultiHeadAttention(
        num_heads=2, key_dim=3, value_dim=5,
        use_bias=True, output_shape=None, attention_axes=None)(inputs[49], inputs[50], inputs[51]))

    shared_conv = Conv2D(1, (1, 1),
                         padding='valid', name='shared_conv', activation='relu')

    up_scale_2 = UpSampling2D((2, 2))
    x1 = shared_conv(up_scale_2(inputs[23]))  # (1, 8, 8)
    x2 = shared_conv(up_scale_2(inputs[24]))  # (1, 8, 8)
    x3 = Conv2D(1, (1, 1), padding='valid')(up_scale_2(inputs[24]))  # (1, 8, 8)
    x = Concatenate()([x1, x2, x3])  # (3, 8, 8)
    outputs.append(x)

    x = Conv2D(3, (1, 1), padding='same', use_bias=False)(x)  # (3, 8, 8)
    outputs.append(x)
    x = Dropout(0.5)(x)
    outputs.append(x)
    x = Concatenate()([
        MaxPooling2D((2, 2))(x),
        AveragePooling2D((2, 2))(x)])  # (6, 4, 4)
    outputs.append(x)

    x = Flatten()(x)  # (1, 1, 96)
    x = Dense(4, use_bias=False)(x)
    outputs.append(x)
    x = Dense(3)(x)  # (1, 1, 3)
    outputs.append(x)

    outputs.append(Add()([inputs[26], inputs[30], inputs[30]]))
    outputs.append(Subtract()([inputs[26], inputs[30]]))
    outputs.append(Multiply()([inputs[26], inputs[30], inputs[30]]))
    outputs.append(Average()([inputs[26], inputs[30], inputs[30]]))
    outputs.append(Maximum()([inputs[26], inputs[30], inputs[30]]))
    outputs.append(Concatenate()([inputs[26], inputs[30], inputs[30]]))

    # TensorFlow 2.16 no longer puts
    # "inbound_nodes": []
    # for such nested models.
    # todo: Check if the situation resolved with later versions.
    if False:
        intermediate_input_shape = (3,)
        intermediate_in = Input(intermediate_input_shape)
        intermediate_x = intermediate_in
        intermediate_x = Dense(8)(intermediate_x)
        intermediate_x = Dense(5, name='duplicate_layer_name')(intermediate_x)
        intermediate_model = Model(
            inputs=[intermediate_in], outputs=[intermediate_x],
            name='intermediate_model')
        intermediate_model.compile(loss='mse', optimizer='nadam')

        x = intermediate_model(x)[0]  # (1, 1, 5)

        intermediate_model_2 = Sequential(name="intermediate_model_2")
        intermediate_model_2.add(Dense(7, input_shape=(5,)))
        intermediate_model_2.add(Dense(5, name='duplicate_layer_name'))
        intermediate_model_2.compile(optimizer='rmsprop',
                                     loss='categorical_crossentropy')

        x = intermediate_model_2(x)  # (1, 1, 5)

        intermediate_model_3_nested = Sequential(name="intermediate_model_3_nested")
        intermediate_model_3_nested.add(Dense(7, input_shape=(6,)))
        intermediate_model_3_nested.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        intermediate_model_3 = Sequential(name="intermediate_model_3")
        intermediate_model_3.add(Dense(6, input_shape=(5,)))
        intermediate_model_3.add(intermediate_model_3_nested)
        intermediate_model_3.add(Dense(8))
        intermediate_model_3.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        x = intermediate_model_3(x)  # (1, 1, 8)

        x = Dense(3)(x)  # (1, 1, 3)

    shared_activation = Activation('tanh')

    outputs = outputs + [
        Activation('tanh')(inputs[25]),
        Activation('hard_sigmoid')(inputs[25]),
        Activation('selu')(inputs[25]),
        Activation('sigmoid')(inputs[25]),
        Activation('softplus')(inputs[25]),
        Activation('softmax')(inputs[25]),
        Activation('relu')(inputs[25]),
        Activation('relu6')(inputs[25]),
        Activation('swish')(inputs[25]),
        Activation('exponential')(inputs[25]),
        Activation('gelu')(inputs[25]),
        Activation('softsign')(inputs[25]),
        Activation('celu')(inputs[25]),
        Activation('elu')(inputs[25]),
        Activation('exponential')(inputs[25]),
        Activation('gelu')(inputs[25]),
        Activation('hard_tanh')(inputs[25]),
        Activation('linear')(inputs[25]),
        Activation('log_sigmoid')(inputs[25]),
        Activation('silu')(inputs[25]),
        Activation('softmax')(inputs[25]),
        Activation('softsign')(inputs[25]),
        Activation('sparse_plus')(inputs[25]),
        Activation('squareplus')(inputs[25]),
        Activation('tanh')(inputs[25]),
        Activation('tanh_shrink')(inputs[25]),
        LeakyReLU(name="real_LeakyReLU_layer", negative_slope=0.5)(inputs[25]),
        ReLU(name="real_ReLU_layer_1")(inputs[25]),
        ReLU(name="real_ReLU_layer_2", max_value=0.4, negative_slope=1.1, threshold=0.3)(inputs[25]),
        ELU(name="real_ELU_layer")(inputs[25]),
        PReLU(name="real_PReLU_layer_1")(inputs[24]),
        PReLU(name="real_PReLU_layer_2")(inputs[25]),
        PReLU(name="real_PReLU_layer_3")(inputs[26]),
        Softmax(name="real_Softmax_layer")(inputs[25]),
        shared_activation(inputs[25]),
        Activation('linear')(inputs[26]),
        Activation('linear')(inputs[23]),
        activations.celu(inputs[25]),
        activations.elu(inputs[25], alpha=0.71),
        activations.exponential(inputs[25]),
        activations.gelu(inputs[25]),
        activations.hard_shrink(inputs[25], threshold=0.31),
        activations.hard_sigmoid(inputs[25]),
        activations.hard_tanh(inputs[25]),
        activations.leaky_relu(inputs[25], negative_slope=0.31),
        activations.linear(inputs[25]),
        activations.log_sigmoid(inputs[25]),
        activations.log_softmax(inputs[25]),
        activations.relu(inputs[25], negative_slope=0.1, max_value=0.8, threshold=0.3),
        activations.relu6(inputs[25]),
        activations.selu(inputs[25]),
        activations.sigmoid(inputs[25]),
        activations.silu(inputs[25]),
        activations.softmax(inputs[25]),
        activations.soft_shrink(inputs[25], threshold=0.31),
        activations.softplus(inputs[25]),
        activations.softsign(inputs[25]),
        activations.sparse_plus(inputs[25]),
        activations.squareplus(inputs[25], b=3),
        activations.tanh(inputs[25]),
        activations.tanh_shrink(inputs[25]),
        activations.threshold(inputs[25], 0.123, 0.423),
        x,
        shared_activation(x),
    ]

    model = Model(inputs=inputs, outputs=outputs, name='test_model_exhaustive')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 2
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=10)
    return model


def get_test_model_embedding() -> Model:
    """Returns a minimalistic test model for the Embedding and CategoryEncoding layers."""

    input_dims = [
        1023,  # maximum integer value in input data
        255,
        15,
    ]
    input_shapes: list[tuple[int, ...]] = [
        (100,),  # must be single-element tuple (for sequence length)
        (1000,),
        (1,),
    ]
    assert len(input_dims) == len(input_shapes)
    output_dims = [8, 3]  # embedding dimension

    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []
    for k in range(2):
        embedding = Embedding(input_dim=input_dims[k], output_dim=output_dims[k])(inputs[k])
        outputs.append(embedding)

    outputs.append(CategoryEncoding(1024, output_mode='multi_hot', sparse=False)(inputs[0]))
    # No longer working since TF 2.16: https://github.com/tensorflow/tensorflow/issues/65390
    # Error: Value passed to parameter 'values' has DataType float32 not in list of allowed values: int32, int64
    # outputs.append(CategoryEncoding(1024, output_mode='count', sparse=False)(inputs[0]))
    # outputs.append(CategoryEncoding(16, output_mode='one_hot', sparse=False)(inputs[2]))
    # Error: Value passed to parameter 'values' has DataType float32 not in list of allowed values: int32, int64
    # outputs.append(CategoryEncoding(1023, output_mode='multi_hot', sparse=True)(inputs[0]))

    model = Model(inputs=inputs, outputs=outputs, name='test_model_embedding')
    model.compile(loss='mse', optimizer='adam')

    # fit to dummy data
    training_data_size = 2
    data_in = generate_integer_input_data(training_data_size, 0, input_dims, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=1)
    return model


def get_test_model_variable() -> Model:
    """Returns a model with variably shaped input tensors."""

    input_shapes = [
        (None, None, 1),
        (None, None, 3),
        (None, 4),
    ]

    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []

    # same as axis=-1
    outputs.append(Concatenate()([inputs[0], inputs[1]]))
    outputs.append(Conv2D(8, (3, 3), padding='same', activation='elu')(inputs[0]))
    outputs.append(Conv2D(8, (3, 3), padding='same', activation='relu')(inputs[1]))
    outputs.append(GlobalMaxPooling2D()(inputs[0]))
    outputs.append(Reshape((2, -1))(inputs[2]))
    outputs.append(Reshape((-1, 2))(inputs[2]))
    outputs.append(MaxPooling2D()(inputs[1]))
    outputs.append(AveragePooling1D(2)(inputs[2]))

    outputs.append(PReLU(shared_axes=[1, 2])(inputs[0]))
    outputs.append(PReLU(shared_axes=[1, 2])(inputs[1]))
    outputs.append(PReLU(shared_axes=[1, 2, 3])(inputs[1]))
    outputs.append(PReLU(shared_axes=[1])(inputs[2]))

    model = Model(inputs=inputs, outputs=outputs, name='test_model_variable')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 2
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=10)
    return model


def get_test_model_autoencoder() -> Model:
    """Returns a minimal autoencoder test model."""
    input_img = Input(shape=(1,), name='input_img')
    encoded = Identity()(input_img)  # Since it's about testing node connections, this suffices.
    encoder = Model(input_img, encoded, name="encoder")

    input_encoded = Input(shape=(1,), name='input_encoded')
    decoded = Identity()(input_encoded)
    decoder = Model(input_encoded, decoded, name="decoder")

    autoencoder_input = Input(shape=(1,), name='input_autoencoder')
    x = encoder(autoencoder_input)
    autoencodedanddecoded = decoder(x)
    autoencoder = Model(inputs=autoencoder_input, outputs=autoencodedanddecoded, name="autoencoder")
    autoencoder.compile(optimizer='sgd', loss='mse')
    return autoencoder


def get_test_model_sequential() -> Model:
    """Returns a typical (VGG-like) sequential test model."""
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(Permute((3, 1, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Permute((2, 3, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), activation='elu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    # fit to dummy data
    training_data_size = 2
    data_in = [np.random.random(size=(training_data_size, 32, 32, 3))]
    data_out = [np.random.random(size=(training_data_size, 10))]
    model.fit(data_in, data_out, epochs=10)
    return model


def main() -> None:
    """Generate different test models and save them to the given directory."""
    if len(sys.argv) != 3:
        print('usage: [model name] [destination file path]', flush=True)
        sys.exit(1)
    else:
        model_name = sys.argv[1]
        dest_path = sys.argv[2]

        get_model_functions = {
            'exhaustive': get_test_model_exhaustive,
            'embedding': get_test_model_embedding,
            'variable': get_test_model_variable,
            'autoencoder': get_test_model_autoencoder,
            'sequential': get_test_model_sequential,
        }

        if not model_name in get_model_functions:
            print('unknown model name: ', model_name)
            sys.exit(2)

        np.random.seed(0)

        model_func = get_model_functions[model_name]
        model = model_func()
        model.save(dest_path)

        # Make sure models can be loaded again,
        # see https://github.com/fchollet/keras/issues/7682
        model = load_model(dest_path)
        model.summary()
        # plot_model(model, to_file= str(model_name) + '.png', show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    main()
