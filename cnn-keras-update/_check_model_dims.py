"""
Authors: Christoph Koerner, Patrick Wahrmann
"""
import fs
import datetime

from utils import *
from models import *

DATASET_DIR = '../data/imdb-wiki-tiny-dataset'

PARAMS = {
  'name': 'CNN Age',
  'input_shape': (3,112,112),
  'n_classes': [10, 2],
  'n_epochs': 30,
  'batchsize': 64,
  'dropout': 0.4,
  'momentum': 0.9,
  'learning_rate': 1e-1,
  'learning_rate_decay': 1e-2,
  'early_stopping_rounds': 10,
  'l2_reg': 2e-4,
  'use_class_weights': False
}

def main(params):
  model = get_levinet_multi(params)
  print(get_model_summary(model))

if __name__ == '__main__':
  main(PARAMS)