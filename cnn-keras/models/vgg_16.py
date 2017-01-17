from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import *


def get_vgg16(input_shape=(3,224,224), n_classes=10, weights_path=None, dropout=0.5,
              init='glorot_uniform', activation='relu', l2_reg=0.0, fc6=4096, fc7=4096):
  
  model = Sequential()
  
  model.add(InputLayer(input_shape=input_shape))

  # Feature Layers
  model.add(Convolution2D(64, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(Convolution2D(64, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Convolution2D(128, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(Convolution2D(128, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Convolution2D(256, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(Convolution2D(256, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(Convolution2D(256, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Convolution2D(512, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(Convolution2D(512, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(Convolution2D(512, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Convolution2D(512, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(Convolution2D(512, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(Convolution2D(512, 3, 3, border_mode='same', init=init, W_regularizer=l2(l2_reg)))
  model.add(Activation(activation))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  # Classification Layers
  model.add(Flatten())
  model.add(Dense(fc6, activation=activation, W_regularizer=l2(l2_reg)))
  model.add(Dropout(dropout))
  model.add(Dense(fc7, activation=activation, W_regularizer=l2(l2_reg)))
  model.add(Dropout(dropout))

  model.add(Dense(n_classes, activation='softmax', init=init))

  if weights_path:
    model.load_weights(weights_path)

  return model