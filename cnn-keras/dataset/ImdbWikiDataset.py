import os.path as fs
import numpy as np

from .ImageDataset import *


class ImdbWikiDataset(ImageDataset):

  def __init__(self, fdir, split):
    self.fdir = fdir
    self.split = split
    self.data = self.load_data()
    self.labels = []
    self.label_names = []
 
  def load_data(self, i=0):
    # TODO: read in all parts
    if self.split == 'train':
      # 0 - 9
      return np.load(fs.join(self.fdir, self.split + '_data_%02d.npy' % i))
    elif self.split == 'test':
      # 0 - 2
      return np.load(fs.join(self.fdir, self.split + '_data_%02d.npy' % i))
    else:
      return np.load(fs.join(self.fdir, self.split + '_data.npy'))

  def load_labels(self, label_str, i=0):
    # TODO: read in all parts
    if self.split == 'train':
      # 0 - 9
      return np.load(fs.join(self.fdir, self.split + '_data_label_%s_%02d.npy' % (label_str, i)))
    elif self.split == 'test':
      # 0 - 2
      return np.load(fs.join(self.fdir, self.split + '_data_label_%s_%02d.npy' % (label_str, i)))
    else:
      return np.load(fs.join(self.fdir, self.split + '_data_label_%s.npy' % (label_str)))

  @functools.lru_cache(maxsize=64)
  def get_mean(self, per_channel=False):
    """Get the global mean of this dataset. The result value is cached."""
    if per_channel:
      return self.samples().mean(axis=(0,2,3))
    return self.samples().mean()

  @functools.lru_cache(maxsize=64)
  def get_stddev(self, per_channel=False):
    """Get the global stddev of this dataset. The result value is cached."""
    if per_channel:
      return self.samples().std(axis=(0,2,3))
    return self.samples().std()

  def size(self):
    """Returns the size of the dataset (number of images)."""
    return self.data.shape[0]

  def nclasses(self):
    """Returns the number of different classes.
    Class labels start with 0 and are consecutive."""
    return len(self.label_names)

  def classname(self, cid):
    """Returns the name of a class as a string."""
    return self.label_names[cid]

  def sample_classname(self, sid):
    """Returns the classname of a sample"""
    return self.classname(self.sample_class(sid))

  def sample_class(self, sid):
    """Returns the class of a sample"""
    return int(self.labels[sid])

  def sample(self, sid):
    return self.data[sid]

  def samples(self):
    return self.data

  def classes(self):
    return self.labels