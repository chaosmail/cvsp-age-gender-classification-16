"""
Author: Christoph Koerner
Student-ID: 0726266
"""
import functools

from .ClassificationDataset import *


class ImageDataset(ClassificationDataset):
  """A dataset, consisting of multiple samples/images
  and corresponding class labels."""
  
  @functools.lru_cache(maxsize=64)
  def get_mean(self, per_channel=False):
    """Get the global mean of this dataset. The result value is cached."""
    if per_channel:
      return np.array(list(self.samples())).mean(axis=(0,1,2))
    return np.array(list(self.samples())).mean()

  @functools.lru_cache(maxsize=64)
  def get_stddev(self, per_channel=False):
    """Get the global stddev of this dataset. The result value is cached."""
    if per_channel:
      return np.array(list(self.samples())).std(axis=(0,1,2))
    return np.array(list(self.samples())).std()

