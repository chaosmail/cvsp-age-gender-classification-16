from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

loss_function = 'categorical_crossentropy'


# categories ... number of categories
def get_model(learn_rate, categories):
    input_shape = (1, 224, 224)

    # based on vgg16 from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    if loss_function == 'categorical_crossentropy':
        model.add(Dense(categories, activation='softmax'))

        # Note: if you change the following line, also change the corresponding line in LoadModelAndWeights.py
        sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)  # Stochastic gradient descent
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
        return model
    elif loss_function == 'mse':
        model.add(Dense(1, activation='softmax'))

        # Note: if you change the following line, also change the corresponding line in LoadModelAndWeights.py
        sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)  # Stochastic gradient descent
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
        return model
