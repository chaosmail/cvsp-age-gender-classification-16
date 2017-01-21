"""
Authors: Christoph Koerner, Patrick Wahrmann
"""
import fs

from models.googlenet_custom_layers import PoolHelper, LRN

from keras.models import load_model

import utils


test_batchsize = 25

DATASET_DIR = '../data/imdb-wiki-tiny-dataset'
MODEL_DIR = './results/2017-01-21_13:17:10'

# Load latest model
latest_model_path = max( fs.find('*.h5', path=MODEL_DIR), key=fs.ctime)
latest_model = load_model(latest_model_path, custom_objects={'LRN':LRN})

# Remove slashes from the layer names
for layer in latest_model._collected_trainable_weights:
  layer.name = layer.name.replace('/', '')
for layer in latest_model.layers:
  layer.name = layer.name.replace('/', '')
# for layer in latest_model.layers_by_depth:
#   layer[0].name = layer[0].name.replace('/', '')
for layer in latest_model.weights:
  layer.name = layer.name.replace('/', '')

# Visualize model
# utils.visualise_with_quiver(load_model(latest_model), '/home/patrick', class_type='gender')
gender = False
if gender:
  utils.visualise_with_quiver(latest_model, class_type='gender')
else:
  utils.visualise_with_quiver(latest_model, class_type='age')