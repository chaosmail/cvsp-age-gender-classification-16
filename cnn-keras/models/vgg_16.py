from keras.models import Sequential
from keras.layers import *


def get_vgg16(input_shape=(3,224,224), n_classes=10, weights_path=None, init='glorot_uniform'):
  
  model = Sequential()
  
  # Feature Layers
  model.add(ZeroPadding2D((1,1),input_shape=input_shape))
  model.add(Convolution2D(64, 3, 3, activation='relu', init=init))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(64, 3, 3, activation='relu', init=init))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128, 3, 3, activation='relu', init=init))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128, 3, 3, activation='relu', init=init))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu', init=init))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu', init=init))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu', init=init))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  # model.add(ZeroPadding2D((1,1)))
  # model.add(Convolution2D(512, 3, 3, activation='relu', init=init))
  # model.add(ZeroPadding2D((1,1)))
  # model.add(Convolution2D(512, 3, 3, activation='relu', init=init))
  # model.add(ZeroPadding2D((1,1)))
  # model.add(Convolution2D(512, 3, 3, activation='relu', init=init))
  # model.add(MaxPooling2D((2,2), strides=(2,2)))

  # model.add(ZeroPadding2D((1,1)))
  # model.add(Convolution2D(512, 3, 3, activation='relu', init=init))
  # model.add(ZeroPadding2D((1,1)))
  # model.add(Convolution2D(512, 3, 3, activation='relu', init=init))
  # model.add(ZeroPadding2D((1,1)))
  # model.add(Convolution2D(512, 3, 3, activation='relu', init=init))
  # model.add(MaxPooling2D((2,2), strides=(2,2)))

  # Classification Layers
  model.add(Flatten())
  # model.add(Dense(4096, activation='relu'))
  # model.add(Dropout(0.5))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))

  model.add(Dense(n_classes, activation='softmax', init=init))

  if weights_path:
    model.load_weights(weights_path)

  return model