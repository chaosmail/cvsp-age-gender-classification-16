"""
Authors: Christoph Koerner, Patrick Wahrmann
"""
import fs
import datetime

from utils import *
from models import *

PARAMS = {
  'name': 'CNN Age',
  'input_shape': (3,227,227),
  'n_classes': 8,
  'n_epochs': 30,
  'batchsize': 64,
  'dropout': 0.4,
}

def main(params):
  model = get_levinet(params)
  print(get_model_summary(model))

if __name__ == '__main__':
  main(PARAMS)