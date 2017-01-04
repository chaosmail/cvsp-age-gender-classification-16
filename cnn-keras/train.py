"""
Authors: Christoph Koerner, Patrick Wahrmann
"""
import fs
import datetime

from dataset import TinyImdbWikiGenderDataset as Dataset
from transformation import get_normalization_transform
from MiniBatchGenerator import MiniBatchGenerator

from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop

import utils
import models


# Configurations
n_epochs = 40
train_batchsize = 40
val_batchsize = 40
test_batchsize = 40

learning_rate = 0.005
decay_rate = 0.999
early_stopping_rounds = 10

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

MODEL_NAME = 'SIMPLE_CNN_GENDER_3_48_48'

DATASET_DIR = '../data/imdb-wiki-tiny-dataset'
MODEL_DIR = '../data/models'
LOGS_DIR = '../data/logs'

MODEL_PATH = fs.join(MODEL_DIR, '%s_%s.h5' % (MODEL_NAME, timestamp))
LOGS_PATH = fs.join(LOGS_DIR, '%s_%s.txt' % (MODEL_NAME, timestamp))

print("Loading %s ..." % Dataset.__class__.__name__)
# Initialize the datasets
ds_train = Dataset(DATASET_DIR, 'train')
ds_val = Dataset(DATASET_DIR, 'val')

# Initialize the preprocessing pipeline
print("Setting up preprocessing ...")
tform = get_normalization_transform(
  means=ds_train.get_mean(per_channel=True),
  stds=ds_train.get_stddev(per_channel=True)
)

# Initialize the MiniBatch generators
print("Initializing minibatch generators ...")
mb_train = MiniBatchGenerator(ds_train, train_batchsize, tform)
mb_val = MiniBatchGenerator(ds_val, val_batchsize, tform)

print(" [%s] %i samples, %i minibatches of size %i" % (
  'train', mb_train.dataset.size(), mb_train.nbatches(), mb_train.batchsize()))
print(" [%s] %i samples, %i minibatches of size %i" % (
  'val', mb_val.dataset.size(), mb_val.nbatches(), mb_val.batchsize()))

# Load the model
print("Initializing model %s ..." % MODEL_NAME)
model = models.get_simple_cnn(
  input_shape=(3,48,48),
  n_classes=2,
  init='glorot_normal'
)

# Initialize optimizer
opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=decay_rate)
print("Initializing %s optimizer ..." % opt.__class__.__name__)
print(" [learning_rate]: %f" % learning_rate)
print(" [rho]: %f" % 0.9)
print(" [decay_rate]: %f" % decay_rate)

model.compile(
  # Multi class classification loss
  loss='categorical_crossentropy',

  # ADAM optimization
  optimizer=opt,
  
  # Add a metric to be evaluated after every batch
  metrics=['accuracy']
)

utils.visualise_with_quiver(model, DATASET_DIR, class_type='age')

# Train model
print("Training for %i epochs ..." % n_epochs)
utils.train(model, mb_train, mb_val, n_epochs,
            best_model_path=MODEL_PATH, logs_path=LOGS_PATH, early_stopping_rounds=early_stopping_rounds)

# Initialize test data
ds_test = Dataset(DATASET_DIR, 'test')
mb_test = MiniBatchGenerator(ds_test, test_batchsize, tform)

print(" [%s] %i samples, %i minibatches of size %i" % (
  ds_test.split, mb_test.dataset.size(), mb_test.nbatches(), mb_test.batchsize()))

# Test best model
print("Testing best model on test set ...")
utils.test(load_model(MODEL_PATH), mb_test)