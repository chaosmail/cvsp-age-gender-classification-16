"""
Authors: Christoph Koerner, Patrick Wahrmann
"""
import fs
import datetime

from dataset import ImdbWikiMultiDataset as Dataset
from MiniBatchGenerator import MiniBatchMultiLossGenerator

from keras.models import load_model
from keras.optimizers import SGD, Adam, RMSprop
from models.googlenet_custom_layers import PoolHelper, LRN

from utils import *
from models import *

DATASET_DIR = '/data/imdb-wiki-dataset'

# Configurations
PARAMS = {
  'name': 'CNN Age Gender',
  'input_shape': (3,112,112),
  'n_classes': [10, 2],
  'n_epochs': 100,
  'batchsize': 64,
  'dropout': 0.5,
  'learning_rate': 1e-2,
  'learning_rate_decay': 1e-3,
  'early_stopping_rounds': 10,
  'l2_reg': 2e-4,
  'activation': 'elu',
  'init': 'glorot_normal',
  'augmentation': True,
  'multi_loss': True
}

def main(params):

  timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
  params['timestamp'] = timestamp

  add_to_report('# Training Report', params)
  add_to_report('\n## Parameters', params)
  add_to_report(params, params)

  print("Loading Dataset ...")
  # Initialize the datasets
  ds_train = Dataset(DATASET_DIR, 'train')
  ds_val = Dataset(DATASET_DIR, 'val')

  # Get the class names for this dataset
  class_names = ds_train.label_names

  mb_train, mb_val, tform = get_train_mb(ds_train, ds_val, params=params)

  if params.get('use_class_weights', False):
    print("Using class weights")
    class_weight = utils.get_class_weight(class_names, ds_train.classes())
    for c, w in class_weight.items():
      print(" [%s] %f" % (class_names[c], w))
  else:
    class_weight = None

  # Initialize a softmax classifier
  print("Initializing CNN and optimizer ...")
  model = get_levinet_multi(params)

  add_model_to_report(model, params)

  # SGD with Nesterov Momentum
  learning_rate = params.get('learning_rate', 1e-2)
  momentum = params.get('momentum', 0.9)
  opt = SGD(lr=learning_rate)

  model.compile(loss={
    'prob1': 'categorical_crossentropy',
    'prob2': 'binary_crossentropy'
  }, optimizer=opt, metrics=['accuracy'])

  try:
    print("Training for %i epochs ..." % params.get('n_epochs'))
    train_loop(model, mb_train, mb_val, opt=opt, params=params)

  except KeyboardInterrupt:
    print("\n\n*** Stopping Training\n\n")
  
  print("Testing best model on test set ...")

  # Initialize test data
  batchsize = params.get('batchsize', 64)

  ds_test = Dataset(DATASET_DIR, 'test')
  mb_test = MiniBatchMultiLossGenerator(ds_test, batchsize, tform)
  print(" [%s] %i samples, %i minibatches of size %i" % (
    ds_test.split, mb_test.dataset.size(), mb_test.nbatches(), mb_test.batchsize()))

  # Load the global best model
  model = load_model('results/%s/best_model.h5' % timestamp,  custom_objects={"LRN": LRN})

  test_loop(model, mb_test, params=params)


if __name__ == '__main__':
  main(PARAMS)