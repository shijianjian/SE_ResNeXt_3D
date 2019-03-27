'''
This code modified from 2d se-resnext
https://github.com/taki0112/SENet-Tensorflow/blob/master/SE_ResNeXt.py

Note:
The first layer was changed to ResNet-like structure due to high memory cost.
Before:
  x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
  x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
 After:
  x = Conv3D(filters=self.init_filters, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same')(x)
  x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(x)
'''
from tensorflow.python.keras.layers import (
    Input,
    GlobalAveragePooling3D,
    Dense,
    AveragePooling3D,
    MaxPooling3D,
    Conv3D,
    BatchNormalization,
    Concatenate,
    Activation,
    Flatten,
    Add,
    Multiply,
    Reshape,
    Lambda,
)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
import tensorflow as tf


class SE_ResNeXt(object):
    def __init__(self, cardinality=8, blocks=3, depth=64, reduction_ratio=4, init_filters=64, training=True):
        """
        The total number of layers is (3*blokcs)*residual_layer_num + 2
        because, blocks = split(conv 2) + transition(conv 1) = 3 layer
        and, first conv layer 1, last dense layer 1
        thus, total number of layers = (3*blocks)*residual_layer_num + 2

        cardinality = 8  : the number of splits
        blocks = 3       : res_block ! (split + transition)
        depth = 64       : out channel

        """
        self.cardinality = cardinality
        self.blocks = blocks
        self.depth = depth
        self.training = training
        self.reduction_ratio = reduction_ratio
        self.init_filters = init_filters

    def first_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Conv3D(filters=self.init_filters, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same')(x)
            x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(x)
            norm = BatchNormalization(axis=-1)(x, training=self.training)
            return Activation("relu")(norm)

    def transform_layer(self, x, strides, scope):
        with tf.name_scope(scope):
            x = Conv3D(filters=self.depth, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
            x = BatchNormalization(axis=-1)(x, training=self.training)
            x = Activation("relu")(x)

            x = Conv3D(filters=self.depth, kernel_size=(3, 3, 3), strides=strides, padding='same')(x)
            x = BatchNormalization(axis=-1)(x, training=self.training)
            x = Activation("relu")(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = Conv3D(filters=out_dim, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x)
            x = BatchNormalization(axis=-1)(x, training=self.training)
            # x = Activation("relu")(x)

            return x

    def split_layer(self, input_x, strides, layer_name):
        with tf.name_scope(layer_name):
            layers_split = list()
            for i in range(self.cardinality):
                splits = self.transform_layer(input_x, strides=strides, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)
            return Concatenate(axis=-1)(layers_split)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):

            squeeze = GlobalAveragePooling3D()(input_x)

            excitation = Dense(units=out_dim / ratio)(squeeze)
            excitation = Activation("relu")(excitation)
            excitation = Dense(units=out_dim)(excitation)
            excitation = Activation("sigmoid")(excitation)
            excitation = Reshape([1, 1, 1, out_dim])(excitation)
            scale = Multiply()([input_x, excitation])

            return scale

    def residual_layer(self, input_x, out_dim, layer_num, res_block=None):
        # split + transform(bottleneck) + transition + merge
        # input_dim = input_x.get_shape().as_list()[-1]
        if res_block is None:
            res_block = self.blocks

        for i in range(res_block):
            input_dim = input_x.get_shape().as_list()[-1]

            if input_dim * 2 == out_dim:
                flag = True
                strides = (2, 2, 2)
                channel = input_dim // 2
            else:
                flag = False
                strides = (1, 1, 1)

            x = self.split_layer(input_x, strides=strides, layer_name='split_layer_' + layer_num + '_' + str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_' + layer_num + '_' + str(i))
            x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=self.reduction_ratio, layer_name='squeeze_layer_' + layer_num + '_' + str(i))
            if flag is True:
                pad_input_x = AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(input_x)
                pad_input_x = Lambda(lambda _x: tf.pad(_x, [[0, 0], [0, 0], [0, 0], [0, 0], [channel, channel]]), output_shape=x.get_shape().as_list())(pad_input_x)  # [?, height, width, channel]
            else:
                pad_input_x = input_x

            input_x = Add()([x, pad_input_x])
            input_x = Activation('relu')(input_x)
        return input_x

    def extract_feature(self, repetitions=3):

        def f(input_x):
            x = self.first_layer(input_x, scope='first_layer')
            features = []
            filters = self.init_filters
            for i in range(1, repetitions + 1):
                print('Building ... %d/%d' % (i, repetitions))
                x = self.residual_layer(x, out_dim=filters, layer_num=str(i))
                features.append(x)
                filters *= 2
            return features

        return f

    def build(self, input_shape, num_output, repetitions=3):
        input_x = Input(shape=input_shape)

        x = self.extract_feature(repetitions=repetitions)(input_x)[-1]
        x = GlobalAveragePooling3D()(x)
        x = Flatten()(x)

        x = Dense(units=num_output,
                  name='final_fully_connected',
                  kernel_initializer="he_normal",
                  kernel_regularizer=l2(1e-4),
                  activation='softmax')(x)

        return Model(inputs=input_x, outputs=x)

if __name__ == '__main__':
    model = SE_ResNeXt(cardinality=3, blocks=3, depth=16, reduction_ratio=4, init_filters=8,
                       training=True).build(input_shape=(200, 1024, 200, 1), num_output=2, repetitions=5)
    print(model.summary())
