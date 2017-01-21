"""
Authors: Christoph Koerner, Patrick Wahrmann
"""
import fs

from models.googlenet_custom_layers import PoolHelper, LRN

from keras.models import load_model

import utils


test_batchsize = 25

DATASET_DIR = '../data/imdb-wiki-tiny-dataset'
MODEL_DIR = './results/2017-01-20_12:58:44'

# Load latest model
latest_model = max( fs.find('*.h5', path=MODEL_DIR), key=fs.ctime)

# Visualize model
# utils.visualise_with_quiver(load_model(latest_model), '/home/patrick', class_type='gender')
gender = False
if gender:
  utils.visualise_with_quiver(load_model(latest_model, custom_objects={'LRN':LRN}), class_type='gender')
else:
  utils.visualise_with_quiver(load_model(latest_model, custom_objects={'LRN':LRN}), class_type='age')