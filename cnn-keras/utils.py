"""
Author: Christoph Koerner
Student-ID: 0726266
"""
from multiprocessing import Process, Queue
import itertools
import numpy as np
import matplotlib.pyplot as plt
import timeit

try:
  import cPickle as pickle
except:
  import pickle

import PIL
from PIL import Image

import fs

# Resampling filter
# PIL.Image.LANCZOS - a high-quality downsampling filter
# PIL.Image.BICUBIC - cubic spline interpolation
RESIZE_TYPE = PIL.Image.LANCZOS


def train(model, mb_train, mb_val, n_epochs, best_model_path, logs_path, early_stopping_rounds=-1):
  from keras.utils import np_utils

  best_val_acc = 0.0
  epoch_of_best_val_acc = -1
  epochs_since_best_val = -1

  logs = np.zeros((n_epochs, 4))

  for epoch in range(n_epochs):

    # Measure duration
    start = timeit.default_timer()

    epochs_since_best_val += 1
    train_loss = np.zeros((mb_train.nbatches()))
    train_acc = np.zeros((mb_train.nbatches()))
    val_loss = np.zeros((mb_val.nbatches()))
    val_acc = np.zeros((mb_val.nbatches()))

    # Shuffle the training batches
    mb_train.shuffle()

    for i in range(mb_train.nbatches()):
      X_batch, y, ids = mb_train.batch(i)
      Y_batch = np_utils.to_categorical(y, mb_train.dataset.nclasses())
      model.train_on_batch(X_batch, Y_batch)
      loss_and_metrics = model.test_on_batch(X_batch, Y_batch)
      train_loss[i] = loss_and_metrics[0]
      train_acc[i] = loss_and_metrics[1]

    for i in range(mb_val.nbatches()):
      X_val, y, ids = mb_val.batch(i)
      Y_val = np_utils.to_categorical(y, mb_val.dataset.nclasses())
      loss_and_metrics = model.test_on_batch(X_val, Y_val)
      val_loss[i] = loss_and_metrics[0]
      val_acc[i] = loss_and_metrics[1]

    stop = timeit.default_timer()

    logs[epoch] = np.array([train_loss.mean(), train_acc.mean(), val_loss.mean(), val_acc.mean()])

    print(" [Epoch %03d] duration: %.1fs, loss: %.3f, training accuracy: %.3f, validation accuracy: %.3f" % (
      epoch, stop - start, train_loss.mean(), train_acc.mean(), val_acc.mean()))

    if val_acc.mean() > best_val_acc:
      best_val_acc = val_acc.mean()
      epoch_of_best_val_acc = epoch
      epochs_since_best_val = 0
      print("  New best validation accuracy, saving model to \"%s\"" % best_model_path)
      model.save(best_model_path)

    elif early_stopping_rounds > 0 and epochs_since_best_val >= early_stopping_rounds:
      print("  Validation accuracy did not improve for %i epochs, stopping" % epochs_since_best_val)
      break

  # Save the logs to disk
  np.save(logs_path, logs)

def test(model, mb_test):
  from keras.utils import np_utils

  # Test the global best model
  test_loss = np.zeros((mb_test.nbatches()))
  test_acc = np.zeros((mb_test.nbatches()))

  for i in range(mb_test.nbatches()):
    X_test, y, ids = mb_test.batch(i)
    Y_test = np_utils.to_categorical(y, mb_test.dataset.nclasses())
    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=len(X_test), verbose=0)
    test_loss[i] = loss_and_metrics[0]
    test_acc[i] = loss_and_metrics[1]

  print(" Accuracy: %.1f%%" % (100*test_acc.mean()))

def sizeof_fmt(num, suffix='B'):
  for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
    if abs(num) < 1024.0:
      return "%3.1f%s%s" % (num, unit, suffix)
    num /= 1024.0
  return "%.1f%s%s" % (num, 'Yi', suffix)

def get_img_array(meta_data, data_src, age_classes, img_dim=(3,224,224), split=0, num_samples_per_split=100000, dtype=np.float32):
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

    # Replace all non valid gender labels with male labels
    y_gender[i - i_start] = meta_data['gender'][i] if int(meta_data['gender'][i]) in [0, 1] else 1
    abspath = fs.join(data_src, meta_data['path'][i])

    # Catch errors
    try:
      with Image.open(abspath) as img:
        img = img.resize(img_dim[1:3], RESIZE_TYPE).convert('RGB')
        X[i - i_start] = np.asarray(img, dtype=dtype).transpose((2,1,0)) / 255
    except OSError as e:
      print("Error reading file %s" % abspath)
      continue
  
  return X, y_age, y_gender

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
  np.random.shuffle(idx)
  return filter_meta_data(data, idx)

def split_meta_data(data, f=0.1):
  intermediate = int(float(len(data['age'])) * f)
  idx1 = np.arange(intermediate)
  idx2 = np.arange(intermediate, len(data['age']))
  return filter_meta_data(data, idx1), filter_meta_data(data, idx2)

def load_meta_data(data_src, wiki_src, imdb_src):
  with open(fs.join(data_src, wiki_src), 'rb') as file:
    wiki_meta = pickle.load(file)
    
  with open(fs.join(data_src, imdb_src), 'rb') as file:
    imdb_meta = pickle.load(file)
  
  return merge_meta_data(imdb_meta, wiki_meta)

# [sudo] pip3 install quiver_engine
def visualise_with_quiver(model, input_images='../data/imdb-wiki-tiny-dataset', class_type='age'):
  classes = ['1-15', '16-20', '21-25', '26-30', '31-35', '36-40', '40-45', '46-50', '51-55', '56-100']
  if class_type != 'age':
    classes = ['female', 'male']

  from quiver_engine import server
  server.launch(
    model,  # a Keras Model

    classes,  # list of output classes from the model to present (if not specified 1000 ImageNet classes will be used)

    5,  # number of top predictions to show in the gui (default 5)

    # where to store temporary files generatedby quiver (e.g. image files of layers)
    temp_folder='./tmp',

    # a folder where input images are stored
    input_folder=input_images,

    # the localhost port the dashboard is to be served on
    port=5000
  )

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(10,10)):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      
  plt.figure(figsize=figsize)
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, '%.2f' % cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

def get_class_weight(labels, y):
  class_perc = {}
  for i in range(len(labels)):
      class_perc[i] = np.sum(y == i) / len(y)

  class_weights = {}
  class_sorted = sorted(class_perc, reverse=True)
  for i in class_sorted:
      class_weights[i] = class_perc[class_sorted[0]] / class_perc[i]

  return class_weights

def run_in_separate_process(method, args):
    def queue_wrapper(q, params):
        r = method(*params)
        q.put(r)

    q = Queue()
    p = Process(target=queue_wrapper, args=(q, args))
    p.start()
    return_val = q.get()
    p.join()
    return return_val

