"""
Authors: Christoph Koerner, Patrick Wahrmann
"""
import fs

from models.googlenet_custom_layers import PoolHelper, LRN

from keras.models import load_model

import utils


test_batchsize = 25

DATASET_DIR = '../data/imdb-wiki-tiny-dataset'

gender = False

if gender:
  latest_model_path = '../models/VGG_16_AGE_3_112_112_2017-01-17_20:09:49.h5'
else:
  # latest_model_path = 'results/2017-01-21_13:17:10/best_model.h5'
  latest_model_path = 'results/2017-01-24_23:55:34/best_model.h5'

# MODEL_DIR = 'results/2017-01-21_13:17:10'
# MODEL_DIR = '../models'

# Load latest model
# latest_model_path = max( fs.find('*.h5', path=MODEL_DIR), key=fs.ctime)
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

if gender:
  utils.visualise_with_quiver(latest_model, class_type='gender', input_images='../data/imdb_crop/00')
else:
  utils.visualise_with_quiver(latest_model, class_type='age', input_images='../data/imdb_crop/00')