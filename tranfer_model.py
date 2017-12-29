import numpy as np
import audio_processor as ap
import keras
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras import backend as K

import tensorflow as tf
from tensorflow import nn as tfnn

model_path = "./model/stupid_model_tensorflow.h5"
style_file = "style.au"
content_file = "content.au"

channel_axis = 3
freq_axis = 1
time_axis = 2

features = keras.models.load_model(model_path)


def calculate_features():
    style_input = np.transpose(ap.get_spectro(style_file), [0, 2, 3, 1])
    content_input = np.transpose(ap.get_spectro(content_file), [0, 2, 3, 1])

    input_layer = features.input
    style_layer = features.layers[8].output
    content_layer = features.layers[16].output
    # content_layer2 = features.layers[-4].output

    print(style_layer)
    print(content_layer)

    sess = K.get_session()

    style_data = sess.run(style_layer, {input_layer: style_input, K.learning_phase(): 0})
    content_data = sess.run(content_layer, {input_layer: content_input, K.learning_phase(): 0})
    # content_data2 = sess.run(content_layer2, {input_layer: content_input, K.learning_phase(): 0})

    style_const = tf.constant(style_data)
    content_const = tf.constant(content_data)
    # content_const2 = tf.constant(content_data2)

    return style_const, content_const


def copy_model():
    initial = tf.random_normal([1, ap.N_FFT2, ap.FRAMES, 1]) * 0.256
    target_data = tf.Variable(initial, dtype=tf.float32, name="result")

    layer = BatchNormalization(axis=freq_axis, name='bn_0_freq_work')
    x = layer(target_data)
    layer.set_weights(features.layers[1].get_weights())
    # Conv block 1
    layer = Convolution2D(64, (3, 3), padding='same', name='conv1_work')
    x = layer(x)
    layer.set_weights(features.layers[2].get_weights())
    layer = BatchNormalization(axis=channel_axis, name='bn1_work')
    x = layer(x)
    layer.set_weights(features.layers[3].get_weights())
    style_out = ELU(name='elu1_work')(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1_work')(style_out)

    # Conv block 2
    layer = Convolution2D(128, (3, 3), padding='same', name='conv2_work')
    x = layer(x)
    layer.set_weights(features.layers[6].get_weights())
    layer = BatchNormalization(axis=channel_axis, name='bn2_work')
    x = layer(x)
    layer.set_weights(features.layers[7].get_weights())
    style_out = ELU(name='elu2_work')(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2_work')(style_out)

    # Conv block 3
    layer = Convolution2D(128, (3, 3), padding='same', name='conv3_work')
    x = layer(x)
    layer.set_weights(features.layers[10].get_weights())
    layer = BatchNormalization(axis=channel_axis, name='bn3_work')
    x = layer(x)
    layer.set_weights(features.layers[11].get_weights())
    x = ELU(name='elu3_work')(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3_work')(x)

    # Conv block 4
    layer = Convolution2D(128, (3, 3), padding='same', name='conv4_work')
    x = layer(x)
    layer.set_weights(features.layers[14].get_weights())
    layer = BatchNormalization(axis=channel_axis, name='bn4_work')
    x = layer(x)
    layer.set_weights(features.layers[15].get_weights())
    content_out = ELU(name='elu4_work')(x)
    '''x = MaxPooling2D(pool_size=(3, 5), name='pool4_work')(content_out)

    # Conv block 5
    layer = Convolution2D(64, (3, 3), padding='same', name='conv5_work')
    x = layer(x)
    layer.set_weights(features.layers[18].get_weights())
    layer = BatchNormalization(axis=channel_axis, name='bn5_work')
    x = layer(x)
    layer.set_weights(features.layers[19].get_weights())
    x = ELU(name='elu5_work')(x)'''
    return target_data, style_out, content_out


def build_model():
    f_style, f_content = calculate_features()
    target, d_style, d_content = copy_model()

    style_loss = tfnn.l2_loss(f_style - d_style)
    content_loss = tfnn.l2_loss(f_content - d_content)
    loss = style_loss + 10 * content_loss

    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss, name="optimizer", var_list=[target])

    tf.summary.FileWriter("./log/", graph=tf.get_default_graph())

    return train, loss, target
