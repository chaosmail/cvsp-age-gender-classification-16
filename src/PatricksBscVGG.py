from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD


def get_model(learn_rate, categories):
    model = Sequential()
    # input: 224x224 grey level images
    input_shape = (1, 224, 224)
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape, init="lecun_uniform"))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, init="lecun_uniform"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.08))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', init="lecun_uniform"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init="lecun_uniform"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Convolution2D(128, 3, 3, border_mode='valid', init="lecun_uniform"))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, init="lecun_uniform"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(1024, init="lecun_uniform"))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(categories, init="lecun_uniform"))  # Output shape must match
    model.add(Activation('softmax'))
    # Note: if you change the following line, also change the corresponding line in LoadModelAndWeights.py
    sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)  # Stochastic gradient descent
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    return model
