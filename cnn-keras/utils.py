"""
Author: Christoph Koerner
Student-ID: 0726266
"""
import itertools
import numpy as np
import timeit
import pyprind
import pprint
from transformation import get_normalization_transform
from MiniBatchGenerator import MiniBatchGenerator, MiniBatchMultiLossGenerator
import keras.backend as K
from keras.utils import np_utils
from keras.utils import layer_utils

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

def flip(X):
  return X.transpose((0,3,2,1))

def train_loop(model, mb_train, mb_val, opt, params={}):
  timestamp = params.get('timestamp')
  lr_decay = params.get('learning_rate_decay', 0.0)
  n_epochs = params.get('n_epochs', 20)
  early_stopping_rounds = params.get('early_stopping_rounds', 0)
  multi_loss = params.get('multi_loss', False)

  best_val_acc = 0.0
  epoch_of_best_val_acc = -1
  epochs_since_best_val = -1

  logs = []

  add_to_report('\n## Training', params)

  for epoch in range(n_epochs):

    # Measure duration
    start = timeit.default_timer()

    epochs_since_best_val += 1

    if multi_loss:
      loss_dim = 3
      accs_dim = 2
    else:
      loss_dim = 1
      accs_dim = 1

    train_loss = np.zeros((mb_train.nbatches(), loss_dim))
    train_acc = np.zeros((mb_train.nbatches(), accs_dim))
    val_loss = np.zeros((mb_val.nbatches(), loss_dim))
    val_acc = np.zeros((mb_val.nbatches(), accs_dim))

    # Shuffle the training batches
    mb_train.shuffle()

    bar = pyprind.ProgBar(mb_train.nbatches() + mb_val.nbatches(),
      title=' [Epoch %03d]' % epoch, bar_char='*')

    for i in range(mb_train.nbatches()):
      X_batch, Y_batch, ids = mb_train.batch(i)
      model.train_on_batch(X_batch, Y_batch, class_weight=params.get('class_weight', None))
      loss_and_metrics = model.test_on_batch(X_batch, Y_batch)
      if multi_loss:
        train_loss[i,0] = loss_and_metrics[0]
        train_loss[i,1] = loss_and_metrics[1]
        train_loss[i,2] = loss_and_metrics[2]
        train_acc[i,0] = loss_and_metrics[3]
        train_acc[i,1] = loss_and_metrics[4]

        bar.update(force_flush=True,
          msg=" %i/%i | train loss: %.3f, acc: %.3f/%.3f" % (i, mb_train.nbatches(),
            loss_and_metrics[0], loss_and_metrics[3], loss_and_metrics[4]
        ))
      else:
        train_loss[i] = loss_and_metrics[0]
        train_acc[i] = loss_and_metrics[1]

        bar.update(force_flush=True,
          msg=" %i/%i | train loss: %.3f, acc: %.3f" % (i, mb_train.nbatches(),
            loss_and_metrics[0], loss_and_metrics[1]
        ))

    for i in range(mb_val.nbatches()):
      X_val, Y_val, ids = mb_val.batch(i)
      loss_and_metrics = model.test_on_batch(X_val, Y_val)
      if multi_loss:
        val_loss[i,0] = loss_and_metrics[0]
        val_loss[i,1] = loss_and_metrics[1]
        val_loss[i,2] = loss_and_metrics[2]
        val_acc[i,0] = loss_and_metrics[3]
        val_acc[i,1] = loss_and_metrics[4]

        bar.update(force_flush=True,
          msg=" %i/%i | val loss: %.3f, acc: %.3f/%.3f" % (i, mb_val.nbatches(),
            loss_and_metrics[0], loss_and_metrics[3], loss_and_metrics[4]
        ))
      else:
        val_loss[i] = loss_and_metrics[0]
        val_acc[i] = loss_and_metrics[1]

        bar.update(force_flush=True,
          msg=" %i/%i | val loss: %.3f, acc: %.3f" % (i, mb_val.nbatches(),
            loss_and_metrics[0], loss_and_metrics[1]
        ))

    stop = timeit.default_timer()

    if multi_loss:
      train_loss_mean = train_loss.mean(axis=0)
      train_accs_mean = train_acc.mean(axis=0)
      val_loss_mean = val_loss.mean(axis=0)
      val_acc_mean = val_acc.mean(axis=0)

      logs.append([
        train_loss_mean[0], train_loss_mean[1], train_loss_mean[2],
        train_accs_mean[0], train_accs_mean[1], 
        val_loss_mean[0], val_loss_mean[1], val_loss_mean[2],
        val_acc_mean[0], val_acc_mean[1]
      ])

      print(" [Epoch %03d] duration: %.1fs, loss: %.3f, train acc: %.3f/%.3f, val acc: %.3f/%.3f" % (
        epoch, stop - start, train_loss_mean[0], train_accs_mean[0], train_accs_mean[1],
        val_acc_mean[0], val_acc_mean[1]))

      add_to_report(" [Epoch %03d] duration: %.1fs, loss: %.3f, train acc: %.3f/%.3f, val acc: %.3f/%.3f" % (
        epoch, stop - start, train_loss_mean[0], train_accs_mean[0], train_accs_mean[1],
        val_acc_mean[0], val_acc_mean[1]), params)

      val_acc_track = val_acc_mean[0]

    else:
      val_acc = val_acc.mean(axis=0)

      logs.append([train_loss.mean(), train_acc.mean(), val_loss.mean(), val_acc.mean()])

      print(" [Epoch %03d] duration: %.1fs, loss: %.3f, train acc: %.3f, val acc: %.3f" % (
        epoch, stop - start, train_loss.mean(), train_acc.mean(), val_acc.mean()))

      add_to_report(" - [Epoch %03d] duration: %.1fs, loss: %.3f, train acc: %.3f, val acc: %.3f" % (
        epoch, stop - start, train_loss.mean(), train_acc.mean(), val_acc.mean()), params)

      val_acc_track = val_acc.mean()

    if opt and lr_decay > 0.0:
      
      K.set_value(opt.lr, (1.0 - lr_decay) * K.get_value(opt.lr))
      lr = K.get_value(opt.lr)

      print("  - Setting learning rate to %f" % lr)

    # Save the logs to disk
    np.save('results/%s/logs' % timestamp, logs)

    if val_acc_track > best_val_acc:
      best_val_acc = val_acc_track
      epoch_of_best_val_acc = epoch
      epochs_since_best_val = 0
      model.save('results/%s/best_model.h5' % timestamp)
      print("  - New best validation accuracy, saving model to \"results/%s/best_model.h5\"" % timestamp)
      add_to_report("  - New best validation accuracy, saving model to \"results/%s/best_model.h5\"" % timestamp, params)

    elif early_stopping_rounds and epochs_since_best_val >= early_stopping_rounds:
      print("  - Validation accuracy did not improve for %i epochs, stopping" % epochs_since_best_val)
      add_to_report("  - Validation accuracy did not improve for %i epochs, stopping" % epochs_since_best_val, params)
      add_to_report("  - Stopping at epoch %i" % epoch, params)
      break


def test_loop(model, mb_test, params={}):
  timestamp = params.get('timestamp')
  multi_loss = params.get('multi_loss', False)

  if multi_loss:
    loss_dim = 3
    accs_dim = 2
  else:
    loss_dim = 1
    accs_dim = 1

  # Test the global best model
  test_loss = np.zeros((mb_test.nbatches(), loss_dim))
  test_acc = np.zeros((mb_test.nbatches(), accs_dim))

  bar = pyprind.ProgBar(mb_test.nbatches(), bar_char='*')

  for i in range(mb_test.nbatches()):
    X_test, Y_test, ids = mb_test.batch(i)
    loss_and_metrics = model.test_on_batch(X_test, Y_test)

    if multi_loss:
      test_loss[i,0] = loss_and_metrics[0]
      test_loss[i,1] = loss_and_metrics[1]
      test_loss[i,2] = loss_and_metrics[2]
      test_acc[i,0] = loss_and_metrics[3]
      test_acc[i,1] = loss_and_metrics[4]

      bar.update(force_flush=True,
        msg=" %i/%i | test loss: %.3f, acc: %.3f/%.3f" % (i, mb_test.nbatches(),
          loss_and_metrics[0], loss_and_metrics[3], loss_and_metrics[4]
      ))

    else:
      test_loss[i] = loss_and_metrics[0]
      test_acc[i] = loss_and_metrics[1]

      bar.update(force_flush=True,
        msg=" %i/%i | test loss: %.3f, acc: %.3f" % (i, mb_test.nbatches(),
          loss_and_metrics[0], loss_and_metrics[1]
      ))

  add_to_report("\n## Testing", params)
  
  if multi_loss:
    test_acc_mean = test_acc.mean(axis=0)
    print(" Accuracy: %.1f%%/%.1f%%" % (100*test_acc_mean[0], 100*test_acc_mean[1]))
    add_to_report("Accuracy: %.1f%%/%.1f%%" % (100*test_acc_mean[0],100*test_acc_mean[1]), params)
  else:
    print(" Accuracy: %.1f%%" % (100*test_acc.mean()))
    add_to_report("Accuracy: %.1f%%" % (100*test_acc.mean()), params)

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

# [sudo] pip3 install git+git://github.com/jakebian/quiver.git
def visualise_with_quiver(model, input_images='../data/', class_type='age', port=5000):
  # classes = np.array(['1-15', '16-20', '21-25', '26-30', '31-35', '36-40', '40-45', '46-50', '51-55', '56-100'])
  # if class_type != 'age':
  #   classes = np.array(['female', 'male'])
  classes = ['1-15', '16-20', '21-25', '26-30', '31-35', '36-40', '40-45', '46-50', '51-55', '56-100']
  if class_type != 'age':
    classes = ['female', 'male']

  print("Starting Quiver on port %i" % port)

  from quiver_engine import server
  server.launch(
    model,  # a Keras Model

    classes,  # list of output classes from the model to present (if not specified 1000 ImageNet classes will be used)

    10,  # number of top predictions to show in the gui (default 5)

    # where to store temporary files generated by quiver (e.g. image files of layers)
    temp_folder='./tmp',

    # a folder where input images are stored
    input_folder=input_images,

    # the localhost port the dashboard is to be served on
    port=port
  )

def get_class_weight(labels, y):
  class_perc = {}
  for i in range(len(labels)):
      class_perc[i] = np.sum(y == i) / len(y)

  class_weights = {}
  class_sorted = sorted(class_perc, reverse=True)
  for i in class_sorted:
      class_weights[i] = class_perc[class_sorted[0]] / class_perc[i]

  return class_weights

def get_model_summary(model, line_length=100, positions=[.33, .55, .67, 1.]):
  out = ""
  if hasattr(model, 'flattened_layers'):
    # Support for legacy Sequential/Merge behavior.
    layers = model.flattened_layers
  else:
    layers = model.layers
  if positions[-1] <= 1:
    positions = [int(line_length * p) for p in positions]

  relevant_nodes = getattr(model, 'container_nodes', None)
  to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Connected to']

  def get_row(fields, positions):
    line = ''
    for i in range(len(fields)):
        if i > 0:
            line = line[:-1] + ' '
        line += str(fields[i])
        line = line[:positions[i]]
        line += ' ' * (positions[i] - len(line))
    return str(line) + "\n"

  out += '_' * line_length
  out += '\n'
  out += get_row(to_display, positions)
  out += '=' * line_length
  out += '\n'

  def get_layer_summary(layer):
    """Prints a summary for a single layer.
    # Arguments
        layer: target layer.
    """
    out = ""
    try:
      output_shape = layer.output_shape
    except AttributeError:
      output_shape = 'multiple'
    connections = []
    for node_index, node in enumerate(layer.inbound_nodes):
      if relevant_nodes:
        node_key = layer.name + '_ib-' + str(node_index)
        if node_key not in relevant_nodes:
          # node is node part of the current network
          continue
      for i in range(len(node.inbound_layers)):
        inbound_layer = node.inbound_layers[i].name
        inbound_node_index = node.node_indices[i]
        inbound_tensor_index = node.tensor_indices[i]
        connections.append(inbound_layer + '[' + str(inbound_node_index) + '][' + str(inbound_tensor_index) + ']')

    name = layer.name
    cls_name = layer.__class__.__name__
    if not connections:
      first_connection = ''
    else:
      first_connection = connections[0]
    fields = [name + ' (' + cls_name + ')', output_shape, layer.count_params(), first_connection]
    out += get_row(fields, positions)
    if len(connections) > 1:
      for i in range(1, len(connections)):
        fields = ['', '', '', connections[i]]
        out += get_row(fields, positions)
    return out

  for i in range(len(layers)):
    out += get_layer_summary(layers[i])
    if i == len(layers) - 1:
      out += '=' * line_length
      out += '\n'
    else:
      out += '_' * line_length
      out += '\n'

  trainable_count, non_trainable_count = layer_utils.count_total_params(layers, layer_set=None)

  out += 'Total params: {:,}'.format(trainable_count + non_trainable_count)
  out += '\n'
  out += 'Trainable params: {:,}'.format(trainable_count)
  out += '\n'
  out += 'Non-trainable params: {:,}'.format(non_trainable_count)
  out += '\n'
  out += '_' * line_length
  out += '\n'

  return out

def get_train_mb(ds_train, ds_val, params={}):
  timestamp = params.get('timestamp')
  batchsize = params.get('batchsize', 64)
  multi_loss = params.get('multi_loss', False)
  augmentation = params.get('augmentation', False)
  add_to_report('\n## Transformations', params)

  # Initialize the preprocessing pipeline
  print("Setting up preprocessing ...")
  tform = get_normalization_transform(
    means=ds_train.get_mean(per_channel=True),
    stds=ds_train.get_stddev(per_channel=True),
    augmentation=augmentation
  )

  add_to_report(" - %s [%s] (values: %s)" % (
    tform.get_transformation(1).__class__.__name__, ds_train.split, str(tform.get_transformation(1).values())), params)

  add_to_report(" - %s [%s] (values: %s)" % (
    tform.get_transformation(2).__class__.__name__, ds_train.split, str(tform.get_transformation(2).values())), params)

  add_to_report('\n## Dataset', params)

  # Initialize the MiniBatch generators
  print("Initializing minibatch generators ...")
  if multi_loss:
    mb_train = MiniBatchMultiLossGenerator(ds_train, batchsize, tform)
    mb_val = MiniBatchMultiLossGenerator(ds_val, batchsize, tform)
  else:
    mb_train = MiniBatchGenerator(ds_train, batchsize, tform)
    mb_val = MiniBatchGenerator(ds_val, batchsize, tform)

  print(" [%s] %i samples, %i minibatches of size %i" % (
    'train', mb_train.dataset.size(), mb_train.nbatches(), mb_train.batchsize()))
  print(" [%s] %i samples, %i minibatches of size %i" % (
    'val', mb_val.dataset.size(), mb_val.nbatches(), mb_val.batchsize()))

  add_to_report(" - [%s] %i samples, %i minibatches of size %i" % (
    'train', mb_train.dataset.size(), mb_train.nbatches(), mb_train.batchsize()), params)
  add_to_report(" - [%s] %i samples, %i minibatches of size %i" % (
    'val', mb_val.dataset.size(), mb_val.nbatches(), mb_val.batchsize()), params)

  return mb_train, mb_val, tform

def add_to_report(text, params={}):
  timestamp = params.get('timestamp')

  if not fs.exists('results/%s' % timestamp):
    fs.mkdir('results/%s' % timestamp)

  if isinstance(text, dict):
    with open('results/%s/report.txt' % timestamp, 'at+') as file:
      pprint.pprint(text, stream=file, indent=2)
  else:
    fs.append("results/%s/report.txt" % timestamp, text + '\n')

def add_model_to_report(model, params):
  timestamp = params.get('timestamp')
  shp = params.get('input_shape', (3,112,112))
  batchsize = params.get('batchsize', 64)

  # Write model structure as JSON file
  fs.write('results/%s/model.json' % timestamp, model.to_json())

  # Print the model Shape
  add_to_report('\n## Model architecture')
  model.summary()
  m_def = get_model_summary(model)
  print('\nUsing architecture:')
  print(m_def)

  # Write model dimensions to Report
  add_to_report(m_def, params)