import os.path as fs
import numpy as np

from .ImdbWikiDataset import *


class ImdbWikiMultiDataset(ImdbWikiDataset):
 
  def __init__(self, fdir, split):
    ImdbWikiDataset.__init__(self, fdir, split)

    self.labels = [self.load_labels('age'), self.load_labels('gender')]
    self.label_names = [[
      '[0 - 15]', '[16 - 20]', '[21 - 25]', '[26 - 30]', '[31 - 35]',
      '[36 - 40]', '[41 - 45]', '[46 - 50]', '[51 - 55]', '[56 - 100]'
    ], [
      'female', 'male'
    ]]

  def nclasses_per(self, lid):
    """Returns the number of different classes.
    Class labels start with 0 and are consecutive."""
    return len(self.label_names[lid])

  def nclasses(self):
    """Returns the number of different classes.
    Class labels start with 0 and are consecutive."""
    return [len(label_names) for label_names in self.label_names]

  def classname(self, lid, cid):
    """Returns the name of a class as a string."""
    return self.label_names[lid][cid]

  def sample_classname(self, lid, sid):
    """Returns the classname of a sample"""
    return self.classname(lid, self.sample_class(lid, sid))

  def sample_classnames(self, sid):
    """Returns the classname of a sample"""
    return [self.classname(lid, cid)
      for lid, cid in enumerate(self.sample_classes(sid))]

  def sample_class(self, lid, sid):
    """Returns the class of a sample"""
    return int(self.labels[lid][sid])

  def sample_classes(self, sid):
    """Returns the class of a sample"""
    return [int(label[sid]) for label in self.labels]

  def classes(self):
    return [label.astype(int) for label in self.labels]