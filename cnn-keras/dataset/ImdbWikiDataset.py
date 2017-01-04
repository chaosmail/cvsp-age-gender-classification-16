import os.path as fs
import numpy as np

from .ImageDataset import *


class ImdbWikiDataset(ImageDataset):

  def __init__(self, fdir, split):
    self.fdir = fdir
    self.split = split
    self.labels = []
    self.label_names = []
    self.train_blocks = 5
    self.test_blocks = 2
    self.data = self.load_data()
 
  def load_data(self, i=0):
    if self.split == 'train':
      data = None
      for i in range(self.train_blocks):
        block = np.load(fs.join(self.fdir, self.split + '_data_%02d.npy' % i))
        data = block if data is None else np.concatenate((data, block), axis=0)
      return data
    elif self.split == 'test':
      data = None
      for i in range(self.test_blocks):
        block = np.load(fs.join(self.fdir, self.split + '_data_%02d.npy' % i))
        data = block if data is None else np.concatenate((data, block), axis=0)
      return data
    else:
      return np.load(fs.join(self.fdir, self.split + '_data.npy'))

  def load_labels(self, label_str, i=0):
    if self.split == 'train':
      meta = None
      for i in range(self.train_blocks):
        block = np.load(fs.join(self.fdir, self.split + '_data_label_%s_%02d.npy' % (label_str, i)))
        meta = block if meta is None else np.concatenate((meta, block), axis=0)
      return meta
    elif self.split == 'test':
      meta = None
      for i in range(self.train_blocks):
        block = np.load(fs.join(self.fdir, self.split + '_data_label_%s_%02d.npy' % (label_str, i)))
        meta = block if meta is None else np.concatenate((meta, block), axis=0)
      return meta
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
    return self.labels.astype(int)