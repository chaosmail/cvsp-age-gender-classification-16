import fs
import PIL
from PIL import Image
import numpy as np
import math
import gc

try:
  import cPickle as pickle
except:
  import pickle

DATA_SRC = '/mnt/s3/datastore'
DATA_DST = '/home/ec2-user/data'

WIKI_CROPS_DIR = "wiki_crop"
IMDB_CROPS_DIR = "imdb_crop"

WIKI_META_OBJ = 'wiki_meta.obj'
IMDB_META_OBJ = 'imdb_meta.obj'

TRAIN_DATA_OBJ = 'train_data'
VAL_DATA_OBJ = 'val_data'
TEST_DATA_OBJ = 'test_data'

# Define the number of samples per split
SAMPLES_PER_SPLIT = 100000
INPUT_DIM = (3,112,112)

TRAIN_TEST_SPLIT = 0.8
TRAIN_VAL_SPLIT = 0.9

# Rescale and convert to greyscale
# PIL.Image.LANCZOS - a high-quality downsampling filter
# PIL.Image.BICUBIC - cubic spline interpolation
RESIZE_TYPE = PIL.Image.LANCZOS

age_classes = [(0,15),(16,20),(21,25),(26,30),(31,35),(36,40),(41,45),(46,50),(51,55),(56,100)]

np.random.seed(42)

lmap = lambda f, l: list(map(f, l))
lfilter = lambda f, l: list(filter(f, l))

def merge_meta_data(data1, data2):
  return {
    'path':   np.append(data1['path'], data2['path']),
    'name':   np.append(data1['name'], data2['name']),
    'age':    np.append(data1['age'], data2['age']),
    'gender': np.append(data1['gender'], data2['gender']),
  }

def filter_meta_data(data, filter_mask):
  return {
    'path':   data['path'][filter_mask].reshape(-1),
    'name':   data['name'][filter_mask].reshape(-1),
    'age':    data['age'][filter_mask].reshape(-1),
    'gender': data['gender'][filter_mask].reshape(-1),
  }

def shuffle_meta_data(data):
  idx = np.arange(len(data['age']))
  rand_idx = np.random.shuffle(idx)
  return filter_meta_data(data, rand_idx)

def split_meta_data(data, f=0.1):
  intermediate = int(float(len(data['age'])) * f)
  idx1 = np.arange(intermediate)
  idx2 = np.arange(intermediate, len(data['age']))
  return filter_meta_data(data, idx1), filter_meta_data(data, idx2)

def sizeof_fmt(num, suffix='B'):
  for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
    if abs(num) < 1024.0:
      return "%3.1f%s%s" % (num, unit, suffix)
    num /= 1024.0
  return "%.1f%s%s" % (num, 'Yi', suffix)

def get_img_array(meta_data, img_dim=(3,224,224), split=0, num_samples_per_split=100000, dtype=np.float32):
  i_start = min(split * num_samples_per_split, len(meta_data['path']))
  i_stop = min(split * num_samples_per_split + num_samples_per_split, len(meta_data['path']))
  num_samples = i_stop - i_start
  
  X = np.zeros((num_samples,img_dim[0],img_dim[1],img_dim[2]), dtype=dtype)
  y_age = np.zeros((num_samples))
  y_gender = np.zeros((num_samples))
  
  print('  Allocating %s for dataset with shape (%i,%i,%i,%i)' %
    (sizeof_fmt(X.nbytes), num_samples, img_dim[0], img_dim[1], img_dim[2]))
  
  age_class = lambda x: age_classes.index(next(filter(lambda e: x >= e[0] and x <= e[1], age_classes)))
  
  for i in range(i_start, i_stop):
    y_age[i - i_start] = age_class(meta_data['age'][i])
    y_gender[i - i_start] = meta_data['gender'][i]
    abspath = fs.join(DATA_SRC, meta_data['path'][i])
    
    with Image.open(abspath) as img:
      img = img.resize(img_dim[1:3], RESIZE_TYPE).convert('RGB')
      X[i - i_start] = np.flipud(np.asarray(img, dtype=dtype).reshape((img_dim)) / 255)
  
  return X, y_age, y_gender

def load_meta_data():
  with open(fs.join(DATA_SRC, WIKI_META_OBJ), 'rb') as file:
    wiki_meta = pickle.load(file)
    
  with open(fs.join(DATA_SRC, IMDB_META_OBJ), 'rb') as file:
    imdb_meta = pickle.load(file)
  
  return merge_meta_data(imdb_meta, wiki_meta)


def main():

  if not fs.exists(DATA_DST):
    fs.mkdir(DATA_DST)

  meta_all = shuffle_meta_data(load_meta_data())

  train, test = split_meta_data(meta_all, TRAIN_TEST_SPLIT)
  train, val = split_meta_data(train, TRAIN_VAL_SPLIT)

  # Free the memory
  del meta_all
  gc.collect()

  print("Converting blocks")

  print(" [train] %i Sapmles. %i Blocks required" % (len(train['path']), math.ceil(len(train['path']) / SAMPLES_PER_SPLIT)))

  for i in range(math.ceil(len(train['path']) / SAMPLES_PER_SPLIT)):
    X_train, y_age, y_gender = get_img_array(train, img_dim=INPUT_DIM, split=i, num_samples_per_split=SAMPLES_PER_SPLIT)
    np.save(fs.add_suffix(fs.join(DATA_DST, TRAIN_DATA_OBJ), '_%02d' % i), X_train)
    np.save(fs.add_suffix(fs.join(DATA_DST, TRAIN_DATA_OBJ), '_label_age_%02d' % i), y_age)
    np.save(fs.add_suffix(fs.join(DATA_DST, TRAIN_DATA_OBJ), '_label_gender_%02d' % i), y_gender)
    
    # Remove the array from memory
    del X_train
    del y_age
    del y_gender
    gc.collect()

  print(" [val] %i Sapmles. 1 Block forced" % (len(val['path'])))

  X_val, y_age, y_gender = get_img_array(val, img_dim=INPUT_DIM, num_samples_per_split=len(val['path']))
  np.save(fs.join(DATA_DST, VAL_DATA_OBJ), X_val)
  np.save(fs.add_suffix(fs.join(DATA_DST, VAL_DATA_OBJ), '_label_age'), y_age)
  np.save(fs.add_suffix(fs.join(DATA_DST, VAL_DATA_OBJ), '_label_gender'), y_gender)

  # Remove the array from memory
  del X_val
  del y_age
  del y_gender
  gc.collect()

  print("[test] %i Sapmles. %i Blocks required" % (len(test['path']), math.ceil(len(test['path']) / SAMPLES_PER_SPLIT)))

  for i in range(math.ceil(len(test['path']) / SAMPLES_PER_SPLIT)):
    X_test, y_age, y_gender = get_img_array(test, img_dim=INPUT_DIM, split=i, num_samples_per_split=SAMPLES_PER_SPLIT)
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