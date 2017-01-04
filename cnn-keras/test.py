"""
Authors: Christoph Koerner, Patrick Wahrmann
"""
import fs

from dataset import TinyImdbWikiAgeDataset as Dataset
from transformation import get_normalization_transform
from MiniBatchGenerator import MiniBatchGenerator

from keras.models import load_model

import utils


test_batchsize = 25

DATASET_DIR = '../data/imdb-wiki-tiny-dataset'
MODEL_DIR = '../data/models'

print("Loading %s ..." % Dataset.__class__.__name__)
# Initialize the datasets
ds_train = Dataset(DATASET_DIR, 'train')
ds_test = Dataset(DATASET_DIR, 'test')

# Initialize the preprocessing pipeline
print("Setting up preprocessing ...")
tform = get_normalization_transform(
  means=ds_train.get_mean(per_channel=True),
  stds=ds_train.get_stddev(per_channel=True),
  # scale_to=255
)

# Initialize the MiniBatch generators
print("Initializing minibatch generators ...")
mb_test = MiniBatchGenerator(ds_test, test_batchsize, tform)

print(" [%s] %i samples, %i minibatches of size %i" % (
  ds_test.split, mb_test.dataset.size(), mb_test.nbatches(), mb_test.batchsize()))

# Load latest model
latest_model = max( fs.find('*.h5', path=MODEL_DIR), key=fs.ctime)

# Visualize model
utils.visualise_with_quiver(load_model(latest_model), DATASET_DIR, class_type='age')

# Test best model
print("Testing model %s on test set ..." % latest_model)
utils.test(load_model(latest_model), mb_test)