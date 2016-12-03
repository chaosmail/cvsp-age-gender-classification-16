import json
# import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import numpy as np
import Data as DataManager

settings = json.load(open("config", "r"))
path_to_saved_model = settings["path_to_saved_model"]
path_to_saved_weights = settings["path_to_saved_weights"]
epochs = int(settings["epochs"])
training_batch_size = int(settings["batch_size"])
augmentation_factor = float(settings["augmentation_factor"])  # How many augmented images to use (1.5 -> if 100 training images, 150 augmented images)
learn_rate = float(settings["learn_rate"])
use_data_augmentation = settings["augmentation"] == 'True'

print("Starting program...")

print("Loading training data...")
data = DataManager.Data()


print("Constructing model...")

model = Sequential()
# input: 224x224 grey level images
inputShape = (1, 224, 224)
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 48, 48), init="lecun_uniform"))
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

model.add(Dense(data.train_labels_cat.shape[1], init="lecun_uniform"))  # Output shape must match
model.add(Activation('softmax'))

print("Training with learning rate " + str(learn_rate) + " and augmentation factor " + str(augmentation_factor) + "...")

# Note: if you change the following line, also change the corresponding line in LoadModelAndWeights.py
sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)  # Stochastic gradient descent
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

print("Saving model...")
json_string = model.to_json()
open(path_to_saved_model, 'w+').write(json_string)

print("Training neural net...")
startTime = time.clock()
# real-time data augmentation:
if use_data_augmentation:
    for epoch in range(epochs):
        print("Starting epoch " + str(epoch+1) + "/" + str(epochs))
        epochStartTime = time.clock()
        data.generate_augmented_data(augmentation_factor)
        model.fit(data.augmented_training_images, data.augmented_training_labels, batch_size=training_batch_size,
                  nb_epoch=1, validation_data=(data.val_images, data.val_labels_cat), shuffle=True)
        epochEndTime = time.clock()
        print("Epoch time (incl data augmentation): " + str(epochEndTime - epochStartTime) + "s")

else:  # Use no data augmentation
    model.fit(data.train_images, data.train_labels_cat, batch_size=training_batch_size, nb_epoch=epochs,
              validation_data=(data.val_images, data.val_labels_cat), shuffle=True)  # , callbacks=remote)

endTime = time.clock()
print("Training took "+str(endTime-startTime)+"s.")

print("Saving weights...")
model.save_weights(path_to_saved_weights)  # hdf5 and h5py must be installed

print("PROGRAM FINISHED")
