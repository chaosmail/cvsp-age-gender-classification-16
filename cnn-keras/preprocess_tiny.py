import fs
import numpy as np
import math
import gc
import utils


# DATA_SRC = '/mnt/s3/datastore/imdb-wiki-dataset'
# DATA_DST = '/data/imdb-wiki-tiny-dataset'

DATA_SRC = '../data'
DATA_DST = '../data/imdb-wiki-tiny-dataset'

WIKI_CROPS_DIR = "wiki_crop"
IMDB_CROPS_DIR = "imdb_crop"

WIKI_META_OBJ = 'wiki_meta.obj'
IMDB_META_OBJ = 'imdb_meta.obj'

TRAIN_DATA_OBJ = 'train_data'
VAL_DATA_OBJ = 'val_data'
TEST_DATA_OBJ = 'test_data'

# Define the number of samples per split
train_samples = 5000
val_samples = 1000
test_samples = 1000

INPUT_DIM = (3,112,112)

TRAIN_TEST_SPLIT = 0.8
TRAIN_VAL_SPLIT = 0.9

age_classes = [(0,15),(16,20),(21,25),(26,30),(31,35),(36,40),(41,45),(46,50),(51,55),(56,100)]

np.random.seed(42)

# lmap = lambda f, l: list(map(f, l))
# lfilter = lambda f, l: list(filter(f, l))

def main():

  if not fs.exists(DATA_DST):
    fs.mkdir(DATA_DST)

  meta_all = utils.shuffle_meta_data(utils.load_meta_data(DATA_SRC, WIKI_META_OBJ, IMDB_META_OBJ))

  train, test = utils.split_meta_data(meta_all, TRAIN_TEST_SPLIT)
  train, val = utils.split_meta_data(train, TRAIN_VAL_SPLIT)

  # Free the memory
  del meta_all
  gc.collect()

  print("Converting blocks")

  print(" [train] %i Sapmles" % (train_samples))

  i = 0
  X_train, y_age, y_gender = utils.get_img_array(train, DATA_SRC, age_classes, img_dim=INPUT_DIM, split=i, num_samples_per_split=train_samples)
  np.save(fs.add_suffix(fs.join(DATA_DST, TRAIN_DATA_OBJ), '_%02d' % i), X_train)
  np.save(fs.add_suffix(fs.join(DATA_DST, TRAIN_DATA_OBJ), '_label_age_%02d' % i), y_age)
  np.save(fs.add_suffix(fs.join(DATA_DST, TRAIN_DATA_OBJ), '_label_gender_%02d' % i), y_gender)
    
  # Remove the array from memory
  del X_train
  del y_age
  del y_gender
  gc.collect()

  print(" [val] %i Sapmles" % (val_samples))

  X_val, y_age, y_gender = utils.get_img_array(val, DATA_SRC, age_classes, img_dim=INPUT_DIM, num_samples_per_split=val_samples)
  np.save(fs.join(DATA_DST, VAL_DATA_OBJ), X_val)
  np.save(fs.add_suffix(fs.join(DATA_DST, VAL_DATA_OBJ), '_label_age'), y_age)
  np.save(fs.add_suffix(fs.join(DATA_DST, VAL_DATA_OBJ), '_label_gender'), y_gender)

  # Remove the array from memory
  del X_val
  del y_age
  del y_gender
  gc.collect()

  print("[test] %i Sapmles" % (test_samples))

  i = 0
  X_test, y_age, y_gender = utils.get_img_array(test, DATA_SRC, age_classes, img_dim=INPUT_DIM, split=i, num_samples_per_split=test_samples)
  np.save(fs.add_suffix(fs.join(DATA_DST, TEST_DATA_OBJ), '_%02d' % i), X_test)
  np.save(fs.add_suffix(fs.join(DATA_DST, TEST_DATA_OBJ), '_label_age_%02d' % i), y_age)
  np.save(fs.add_suffix(fs.join(DATA_DST, TEST_DATA_OBJ), '_label_gender_%02d' % i), y_gender)
  
  # Remove the array from memory
  del X_test
  del y_age
  del y_gender
  gc.collect()

if __name__ == '__main__':
  main()