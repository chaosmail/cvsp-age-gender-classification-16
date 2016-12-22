"""
Author: Christoph Koerner
Student-ID: 0726266
"""
import numpy as np
import timeit

from keras.utils import np_utils


def train(model, mb_train, mb_val, n_epochs, best_model_path, logs_path, early_stopping_rounds=-1):
  
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