"""
Authors: Christoph Koerner, Patrick Wahrmann
"""
import os.path as fs
import timeit

from dataset import TinyImdbWikiAgeDataset as Dataset
from dataset import *
from transformation import get_normalization_transform
from MiniBatchGenerator import *

from keras.models import load_model
from keras.optimizers import SGD
from keras.utils import np_utils


# Configurations
n_epochs = 100
train_batchsize = 1
val_batchsize = 1
test_batchsize = 1
momentum = 0.9
learning_rate = 0.001
decay = 0.0001
early_stopping_rounds = 10

DATASET_DIR = '../data/packaged'

print("Loading %s ..." % Dataset.__class__.__name__)
# Initialize the datasets
ds_train = Dataset(DATASET_DIR, 'train')
ds_val = Dataset(DATASET_DIR, 'val')

# Initialize the preprocessing pipeline
print("Setting up preprocessing ...")
tform = get_normalization_transform(
  means=ds_train.get_mean(per_channel=True),
  stds=ds_train.get_stddev(per_channel=True)
)

# Initialize the MiniBatch generators
print("Initializing minibatch generators ...")
mb_train = MiniBatchGenerator(ds_train, train_batchsize, tform)
mb_val = MiniBatchGenerator(ds_val, val_batchsize, tform)
print(" [%s] %i samples, %i minibatches of size %i" % (
  'train', mb_train.dataset.size(), mb_train.nbatches(), mb_train.batchsize()))
print(" [%s] %i samples, %i minibatches of size %i" % (
  'val', mb_val.dataset.size(), mb_val.nbatches(), mb_val.batchsize()))

# Initialize a softmax classifier
print("Initializing CNN and optimizer ...")

# Use VGG 16
from models import VGG_16_AGE_3_112_112 as model
print(" Using VGG 16 with input dimensions 3x112x112")

model.compile(
  # Multi class classification loss
  loss='categorical_crossentropy',

  # SGD with Nesterov Momentum
  optimizer=SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=True),
  
  # Add a metric to be evaluated after every batch
  metrics=['accuracy']
)

print("Training for %i epochs ..." % n_epochs)

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
    loss_and_metrics = model.evaluate(X_batch, Y_batch, batch_size=len(X_batch), verbose=0)
    train_loss[i] = loss_and_metrics[0]
    train_acc[i] = loss_and_metrics[1]

  for i in range(mb_val.nbatches()):
    X_val, y, ids = mb_val.batch(i)
    Y_val = np_utils.to_categorical(y, mb_val.dataset.nclasses())
    loss_and_metrics = model.evaluate(X_val, Y_val, batch_size=len(X_val), verbose=0)
    val_loss[i] = loss_and_metrics[0]
    val_acc[i] = loss_and_metrics[1]

  stop = timeit.default_timer()

  logs[epoch] = np.array([train_loss.mean(), train_acc.mean(), val_loss.mean(), val_acc.mean()])

  print(" [Epoch %03d] duration: %.1fs, loss: %.3f, training accuracy: %.3f, validation accuracy: %.3f" % (
    stop - start, epoch, train_loss.mean(), train_acc.mean(), val_acc.mean()))

  if val_acc.mean() > best_val_acc:
    best_val_acc = val_acc.mean()
    epoch_of_best_val_acc = epoch
    epochs_since_best_val = 0
    model.save('best_model_age.h5')
    print("  New best validation accuracy, saving model to \"best_model_age.h5\"")

  elif epochs_since_best_val >= early_stopping_rounds:
    print("  Validation accuracy did not improve for %i epochs, stopping" % epochs_since_best_val)
    break

print("Testing best model on test set ...")

# Initialize test data
ds_test = Dataset(DATASET_DIR, 'test')
mb_test = MiniBatchGenerator(ds_test, test_batchsize, tform)
print(" [%s] %i samples, %i minibatches of size %i" % (
  ds_test.split, mb_test.dataset.size(), mb_test.nbatches(), mb_test.batchsize()))

# Load the global best model
model = load_model('best_model_age.h5')

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

# Save the logs to disk
np.save('logs/_logs_cnn_classify_cifar10', logs)