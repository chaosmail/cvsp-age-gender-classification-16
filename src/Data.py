import json

import numpy as np
from PIL import Image
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras.preprocessing.image as debug_prepro

settings = json.load(open("config", "r"))
path_to_npz = settings["path_to_npz"]
use_data_augmentation = settings["augmentation"] == 'True'
augmentation_factor = float(settings["augmentation_factor"])  # How many augmented images to use (1.5 -> if 100 training images, 150 augmented images)


class Data:
    usages = None
    labels = None
    images = None
    test_images = None
    test_labels = None
    test_labels_cat = None
    train_images = None
    train_labels = None
    train_labels_cat = None
    val_images = None
    val_labels = None
    val_labels_cat = None
    augmented_data_generator = None
    augmented_training_images = None
    augmented_training_labels = None

    def __init__(self):
        self.load_images_from_npz()

    def load_images_from_npz(self):

        # Loading npz data as in https://www.getdatajoy.com/learn/Read_and_Write_Numpy_Binary_Files

        # Load the npz file (archive file for numpy arrays)
        npz = np.load(path_to_npz)

        # List arrays stored in the file
        # print('arrays stored in this file: \n', npz.files)

        # Extract the arrays 'usages', 'images' and 'labels'
        self.usages = npz['usages']  # 0 = train, 1 = val, 2 = test
        self.images = npz['images']
        self.labels = npz['labels']  # (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

        # Change data format from int to float
        self.images = self.images.astype(dtype=np.float64, copy=False)

        # Normalize images (mean subtraction and unit variance)
        for i in range(0, self.images.shape[0]-1):
            im = self.images[i, :, :]
            im -= np.mean(im)
            std = np.std(im)
            im /= std
            self.images[i, :, :] = im



        # count how much images of which kind are available
        amount = np.array([0, 0, 0])
        for i in range(0, self.usages.shape[0]):
            amount[self.usages[i]] += 1

        # initialize all variables with the calculated sizes
        self.test_images = np.zeros((amount[2], 48, 48), dtype=np.float64)
        self.test_labels = np.zeros((amount[2], 1), dtype=np.int16)
        self.train_images = np.zeros((amount[0], 48, 48), dtype=np.float64)
        self.train_labels = np.zeros((amount[0], 1), dtype=np.int16)
        self.val_images = np.zeros((amount[1], 48, 48), dtype=np.float64)
        self.val_labels = np.zeros((amount[1], 1), dtype=np.int16)

        # fill the variables (divide images into the right types)
        test_counter = 0
        train_counter = 0
        val_counter = 0
        for i in range(0, self.usages.shape[0]):
            if self.usages[i] == 0:  # train
                self.train_images[train_counter, :, :] = self.images[i, :, :]
                self.train_labels[train_counter, 0] = self.labels[i, 0]
                train_counter += 1
            elif self.usages[i] == 1:  # validation
                self.val_images[val_counter, :, :] = self.images[i, :, :]
                self.val_labels[val_counter, 0] = self.labels[i, 0]
                val_counter += 1
            elif self.usages[i] == 2:  # test
                self.test_images[test_counter, :, :] = self.images[i, :, :]
                self.test_labels[test_counter, 0] = self.labels[i, 0]
                test_counter += 1

        # TODO remove:
        # self.show_image(self.train_images[0,:,:])

        # Keras does this instead of me
        # Do mean subtraction (subtract mean and divide through std deviation)
        # Note: Mean and standard deviation are only calculated with training data
        # sum_of_all = np.zeros((48, 48), dtype=np.float64)  # caution: max ~8 million training images, danger of overflow
        # for i in range(0, self.train_images.shape[0]):
        #     sum_of_all += self.train_images[i, :, :]
        # mean = sum_of_all / self.train_images.shape[0]
        # # TODO remove:
        # # self.show_image(mean)
        # sum_of_squared_differences = np.zeros((48, 48), dtype=np.float64)
        # for i in range(0, self.train_images.shape[0]):
        #     sum_of_squared_differences += np.square(self.train_images[i, :, :] - mean)
        # variance = (1.0/(self.train_images.shape[0]-1))*sum_of_squared_differences
        # std_deviation = np.sqrt(variance)
        # self.train_images = ((self.train_images - mean)/std_deviation).astype(np.float32)
        # self.val_images = ((self.val_images - mean)/std_deviation).astype(np.float32)
        # self.test_images = ((self.test_images - mean)/std_deviation).astype(np.float32)
        # TODO remove:
        # self.show_image((self.train_images[0,:,:]+0.5)*255)



        # Keras needs ax1x224x224, not ax224x224 -> reshape. http://stackoverflow.com/q/36232819
        self.train_images = self.train_images.reshape((self.train_images.shape[0], 1, 48, 48))
        self.val_images = self.val_images.reshape((self.val_images.shape[0], 1, 48, 48))
        self.test_images = self.test_images.reshape((self.test_images.shape[0], 1, 48, 48))

        # Keras needs classes as ex. 0000010 instead of 5
        self.train_labels_cat = to_categorical(self.train_labels)
        self.val_labels_cat = to_categorical(self.val_labels)
        self.test_labels_cat = to_categorical(self.test_labels)

        # Keras data augmentation
        if use_data_augmentation:
            self.augmented_data_generator = ImageDataGenerator(
                # BEWARE: bug in implementation (https://github.com/fchollet/keras/issues/2559)
                # samplewise_center=True,
                # samplewise_std_normalization=True,
                rotation_range=5,
                width_shift_range=0.15,
                height_shift_range=0.15,
                zca_whitening=False,
                dim_ordering="th",
                horizontal_flip=True)
            # compute quantities required for featurewise normalization
            self.augmented_data_generator.fit(self.train_images, augment=True, rounds=1)
            # self.generate_augmented_data(augmentation_factor) # this is done before every epoch now

        return self.train_images, self.train_labels

    @staticmethod
    def show_image(image_array):  # if you want to show an image: http://stackoverflow.com/a/33482326
        im = Image.fromarray(image_array)
        im.show()

    @staticmethod
    def save_image(filename, image):
        import scipy.misc
        scipy.misc.toimage(image, cmin=0.0, cmax=255.0).save(filename)

    def calculate_statistics_about_data(self):
        # Training data ---------------------------------------------------
        amounts = np.zeros((7, 1), dtype=np.int16)
        # count occurrences
        for i in range(0, self.train_labels.shape[0]):
            emotion = self.train_labels[i, 0]
            amounts[emotion, 0] += 1

        # print results
        print("Training data: (total size: "+str(self.train_labels.shape[0])+")")
        for i in range(0, amounts.shape[0]):
            occurrences = amounts[i, 0]
            percentage = (float(occurrences)/self.train_labels.shape[0])*100
            print("Emotion #"+str(i)+" occurred " + str(occurrences)+" times. (="+str(percentage)+"%)")

        # Test data -------------------------------------------------------
        amounts = np.zeros((7, 1), dtype=np.int16)
        # count occurrences
        for i in range(0, self.test_labels.shape[0]):
            emotion = self.test_labels[i, 0]
            amounts[emotion, 0] += 1

        # print results
        print("Test data: (total size: "+str(self.test_labels.shape[0])+")")
        for i in range(0, amounts.shape[0]):
            occurrences = amounts[i, 0]
            percentage = (float(occurrences) / self.test_labels.shape[0]) * 100
            print("Emotion #" + str(i) + " occurred " + str(occurrences) + " times. (=" + str(percentage) + "%)")

        # Validation data -------------------------------------------------
        amounts = np.zeros((7, 1), dtype=np.int16)
        # count occurrences
        for i in range(0, self.val_labels.shape[0]):
            emotion = self.val_labels[i, 0]
            amounts[emotion, 0] += 1

        # print results
        print("Validation data: (total size: "+str(self.val_labels.shape[0])+")")
        for i in range(0, amounts.shape[0]):
            occurrences = amounts[i, 0]
            percentage = (float(occurrences) / self.val_labels.shape[0]) * 100
            print("Emotion #" + str(i) + " occurred " + str(occurrences) + " times. (=" + str(percentage) + "%)")

        # All data -------------------------------------------------
        amounts = np.zeros((7, 1), dtype=np.int16)
        # count occurrences
        for i in range(0, self.labels.shape[0]):
            emotion = self.labels[i, 0]
            amounts[emotion, 0] += 1

        # print results
        print("All data: (total size: "+str(self.labels.shape[0])+")")
        for i in range(0, amounts.shape[0]):
            occurrences = amounts[i, 0]
            percentage = (float(occurrences) / self.labels.shape[0]) * 100
            print("Emotion #" + str(i) + " occurred " + str(occurrences) + " times. (=" + str(percentage) + "%)")

    def generate_augmented_data(self, factor):
        print("Generating augmented data...")
        gen = self.augmented_data_generator.flow(self.train_images, self.train_labels_cat,
                                                 batch_size=self.train_images.shape[0],
                                                 shuffle=False)
        self.augmented_training_images, self.augmented_training_labels = gen.__next__()
        for i in range(0, int(factor)-1):
            augmented_images_batch, augmented_labels_batch = gen.__next__()
            self.augmented_training_images = np.concatenate((self.augmented_training_images, augmented_images_batch))
            self.augmented_training_labels = np.concatenate((self.augmented_training_labels, augmented_labels_batch))
