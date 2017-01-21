"""
Authors: Christoph Koerner, Patrick Wahrmann
"""
import fs
import datetime

from dataset import ImdbWikiGenderDataset as Dataset
from transformation import get_normalization_transform
from MiniBatchGenerator import MiniBatchGenerator

from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop

import utils
import models


# Configurations
shp = (3,112,112)

n_epochs = 30
batchsize = 64
learning_rate = 1e-2
decay_rate = 0.0
l2_reg = 1e-3
early_stopping_rounds = 10
use_class_weights = False

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

MODEL_NAME = 'VGG_16_GENDER_%i_%i_%i' % shp

DATASET_DIR = '/data/imdb-wiki-dataset'
MODEL_DIR = '/data/models'
LOGS_DIR = '/data/logs'

MODEL_PATH = fs.join(MODEL_DIR, '%s_%s.h5' % (MODEL_NAME, timestamp))
LOGS_PATH = fs.join(LOGS_DIR, '%s_%s.txt' % (MODEL_NAME, timestamp))

print("Loading Dataset ...")
# Initialize the datasets
ds_train = Dataset(DATASET_DIR, 'train')
ds_val = Dataset(DATASET_DIR, 'val')

# Get the class names for this dataset
class_names = ds_train.label_names

# Initialize the preprocessing pipeline
print("Setting up preprocessing ...")
tform = get_normalization_transform(
  means=ds_train.get_mean(per_channel=True),
  stds=ds_train.get_stddev(per_channel=True)
)

# Initialize the MiniBatch generators
print("Initializing minibatch generators ...")
mb_train = MiniBatchGenerator(ds_train, batchsize, tform)
mb_val = MiniBatchGenerator(ds_val, batchsize, tform)

print(" [%s] %i samples, %i minibatches of size %i" % (
  'train', mb_train.dataset.size(), mb_train.nbatches(), mb_train.batchsize()))
print(" [%s] %i samples, %i minibatches of size %i" % (
  'val', mb_val.dataset.size(), mb_val.nbatches(), mb_val.batchsize()))

if use_class_weights:
  print("Using class weights")
  class_weight = utils.get_class_weight(class_names, ds_train.classes())
  for c, w in class_weight.items():
    print(" [%s] %f" % (class_names[c], w))
else:
  class_weight = None

# Load the model
print("Initializing model %s ..." % MODEL_NAME)
model = models.get_vgg16(
  input_shape=shp, n_classes=len(class_names),
  init='glorot_normal', # Seems to be more stable than glorot uniform (Xavier Initialization)
  #batch_norm=False,      # Better generalization and faster convergence, no improvements with ELU (slower computation)
  l2_reg=l2_reg,        # Adds good generalization to the model
  activation='elu',     # Seems to converge much faster instead of ReLU
  dropout=0.5,          # More generalization, drop rand connections at training time,
  fc6=4096,
  fc7=4096
)

# Print the model Shape
print('Using architecture:')
utils.get_model_shape(model, (batchsize, shp[0], shp[1], shp[2]))

# Initialize optimizer
opt = SGD(lr=learning_rate, decay=decay_rate)
print("Initializing %s optimizer ..." % type(opt).__name__)
print(" [learning_rate]: %f" % learning_rate)
print(" [decay_rate]: %f" % decay_rate)

model.compile(
  # Multi class classification loss
  loss='categorical_crossentropy',

  # ADAM optimization
  optimizer=opt,
  
  # Add a metric to be evaluated after every batch
  metrics=['accuracy']
)

# Use only in testing
# utils.visualise_with_quiver(model, DATASET_DIR, class_type='age')

# Train model
print("Training for %i epochs ..." % n_epochs)
utils.train(model, mb_train, mb_val, n_epochs,
            best_model_path=MODEL_PATH, logs_path=LOGS_PATH, early_stopping_rounds=early_stopping_rounds)

# Initialize test data
ds_test = Dataset(DATASET_DIR, 'test')
mb_test = MiniBatchGenerator(ds_test, batchsize, tform)

print(" [%s] %i samples, %i minibatches of size %i" % (
  ds_test.split, mb_test.dataset.size(), mb_test.nbatches(), mb_test.batchsize()))

# Test best model
print("Testing best model on test set ...")
utils.test(load_model(MODEL_PATH), mb_test)