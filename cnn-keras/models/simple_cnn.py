"""
Author: Christoph Koerner
Student-ID: 0726266
"""
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import *


def get_simple_cnn(params, class_layers=True):

  input_shape = params.get('input_shape', (3,224,224))
  activation = params.get('activation', 'relu')
  l2_reg = params.get('l2_reg', 2e-4)
  dropout = params.get('dropout', 0.5)
  n_classes = params.get('n_classes', 10)
  init = params.get('init', 'glorot_uniform')
  fc6 = params.get('fc6', 1024)
  fc7 = params.get('fc7', 1024)
  batch_norm = params.get('batchnorm', False)

  model = Sequential()
  border_mode = 'valid'
  bn_axis = 1
  bn_mode = 0

  # Input Layer
  model.add(InputLayer(input_shape=input_shape))

  # Feature Layers
  model.add(Convolution2D(96, 3, 3, border_mode=border_mode, init=init,
    W_regularizer=l2(l2_reg), activation=activation))
  if batch_norm:
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
  model.add(MaxPooling2D((2,2), strides=(2,2)))
  
  model.add(Convolution2D(256, 3, 3, border_mode=border_mode, init=init,
    W_regularizer=l2(l2_reg), activation=activation))
  if batch_norm:
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Convolution2D(384, 3, 3, border_mode=border_mode, init=init,
    W_regularizer=l2(l2_reg), activation=activation))
  if batch_norm:
    model.add(BatchNormalization(mode=bn_mode, axis=bn_axis))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Flatten())

  # Classification Layers
  if class_layers:
    model.add(Dense(fc6, init=init, W_regularizer=l2(l2_reg), activation=activation))
    model.add(Dropout(dropout))

    model.add(Dense(fc7, init=init, W_regularizer=l2(l2_reg), activation=activation))
    model.add(Dropout(dropout))

  # Loss Layer
  if n_classes is not None:
    model.add(Dense(output_dim=n_classes, activation='softmax',
      init=init, W_regularizer=l2(l2_reg)))

  return model


def get_shared_cnn(input_shape=(3,224,224), n_classes=[10, 2], fc6=[1000, 1000], fc7=[1000, 1000],
                   dropout=0.5, l2_reg=0.0, activation='relu', init='glorot_uniform', batch_norm=False):
  """Take a look at https://github.com/fchollet/keras/issues/1320 for more information"""

  # Shared feature extraction layer
  model_feat = get_simple_cnn(input_shape, n_classes=None,
    l2_reg=l2_reg, activation=activation, init=init, batch_norm=batch_norm)

  # Combine shared feature extractors with 2 classifier
  model1 = Sequential()
  model1.add(model_feat)
  model1.add(Dense(fc6[0], activation=activation, init=init, W_regularizer=l2(l2_reg)))
  model1.add(Dropout(dropout))
  model1.add(Dense(fc7[0], activation=activation, init=init, W_regularizer=l2(l2_reg)))
  model1.add(Dropout(dropout))

  # Output 1
  model1.add(Dense(output_dim=n_classes[0], activation='softmax',
    init=init, W_regularizer=l2(l2_reg)))

  model2 = Sequential()
  model2.add(model_feat)
  model2.add(Dense(fc6[1], activation=activation, init=init, W_regularizer=l2(l2_reg)))
  model2.add(Dropout(dropout))
  model2.add(Dense(fc7[1], activation=activation, init=init, W_regularizer=l2(l2_reg)))
  model2.add(Dropout(dropout))

  # Output 2
  model2.add(Dense(output_dim=n_classes[1], activation='softmax',
    init=init, W_regularizer=l2(l2_reg)))

  return model1, model2