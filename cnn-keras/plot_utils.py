import numpy as np
import matplotlib.pyplot as plt

from utils import *


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

def plot_perf(hist, hist2=None, metric='loss', figsize=(20,5)):
  plt.figure(figsize=figsize)
  if hist2 is not None:
    x = list(hist.epoch) + list(np.array(hist2.epoch) + np.max(hist1.epoch) + 1)
    y1 = list(hist.history[metric]) + list(hist2.history[metric])
    y2 = list(hist.history['val_%s' % metric]) + list(hist2.history['val_%s' % metric])
  else:
    x = hist.epoch
    y1 = hist.history[metric]
    y2 = hist.history['val_%s' % metric]
  plt.plot(x, y1, label='train')
  plt.plot(x, y2, label='val')
  plt.xlabel('epoch')
  plt.ylabel(metric)
  plt.legend()