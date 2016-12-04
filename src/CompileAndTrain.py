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
import VGG16

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


print("Constructing model...with learning rate " + str(learn_rate) +
      " and augmentation factor " + str(augmentation_factor) + "...")
model = VGG16.get_model(learn_rate, data.train_labels_cat.shape[1])

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
