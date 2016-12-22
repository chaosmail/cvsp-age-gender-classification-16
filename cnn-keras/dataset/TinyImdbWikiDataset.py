import os.path as fs
import numpy as np

from .ImdbWikiDataset import *


class TinyImdbWikiDataset(ImdbWikiDataset):
 
  def load_data(self, i=0):
    # TODO: read in all parts
    if self.split == 'train':
      # 0 - 9
      return np.load(fs.join(self.fdir, self.split + '_data_00.npy'))
    elif self.split == 'test':
      # 0 - 2
      return np.load(fs.join(self.fdir, self.split + '_data_00.npy'))
    else:
      return np.load(fs.join(self.fdir, self.split + '_data.npy'))

  def load_labels(self, label_str, i=0):
    # TODO: read in all parts
    if self.split == 'train':
      # 0 - 9
      return np.load(fs.join(self.fdir, self.split + '_data_label_%s_00.npy' % label_str))
    elif self.split == 'test':
      # 0 - 2
      return np.load(fs.join(self.fdir, self.split + '_data_label_%s_00.npy' % label_str))
    else:
      return np.load(fs.join(self.fdir, self.split + '_data_label_%s.npy' % label_str))