"""
Author: Christoph Koerner
Student-ID: 0726266
"""
import numpy as np
from keras.utils import np_utils


class MiniBatchGenerator:
  # Create minibatches of a given size from a dataset.
  # Preserves the original sample order unless shuffle() is used.

  def __init__(self, dataset, bs, tform=None):
    # Constructor.
    # dataset is a ClassificationDataset to wrap.
    # bs is an integer specifying the minibatch size.
    # tform is an optional SampleTransformation.
    # If given, tform is applied to all samples returned in minibatches.
    self.dataset = dataset
    self.bs = bs
    self.tform = tform
    self.indices = np.arange(dataset.size())

  def batchsize(self):
    # Return the number of samples per minibatch.
    # The size of the last batch might be smaller.
    return self.bs

  def nbatches(self):
    # Return the number of minibatches.
    return int(np.ceil(self.dataset.size() / self.bs))

  def shuffle(self):
    # Shuffle the dataset samples so that each
    # ends up at a random location in a random minibatch.
    np.random.shuffle(self.indices)

  def batch(self, bid, hot_one=True):
    # Return the bid-th minibatch.
    # Batch IDs start with 0 and are consecutive.
    # Throws an error if the minibatch does not exist.
    if bid >= self.nbatches():
      raise ValueError('Cannot retrieve batch %i from %i batches' % (bid, self.nbatches()))

    bs = self.bs
    if self.dataset.size() % self.bs and bid == self.nbatches() - 1:
      bs = self.dataset.size() % self.bs

    st = self.bs * bid
    tf = self.tform.apply
    ix = self.indices[st:st+bs]

    X = np.apply_along_axis(tf, 0, self.dataset.samples()[ix,:]).astype(np.float16)
    
    if hot_one:
      y = np_utils.to_categorical(self.dataset.classes()[ix], self.dataset.nclasses())
    else:
      y = self.dataset.classes()[ix]

    return X, y, ix

class MiniBatchMultiLossGenerator(MiniBatchGenerator):

  def batch(self, bid, hot_one=True):
    # Return the bid-th minibatch.
    # Batch IDs start with 0 and are consecutive.
    # Throws an error if the minibatch does not exist.
    if bid >= self.nbatches():
      raise ValueError('Cannot retrieve batch %i from %i batches' % (bid, self.nbatches()))

    bs = self.bs
    if self.dataset.size() % self.bs and bid == self.nbatches() - 1:
      bs = self.dataset.size() % self.bs

    st = self.bs * bid
    tf = self.tform.apply
    ix = self.indices[st:st+bs]

    X = np.apply_along_axis(tf, 0, self.dataset.samples()[ix,:]).astype(np.float16)

    if hot_one:
      y = [np_utils.to_categorical(yb[ix].astype(int), self.dataset.nclasses_per(i))
            for i, yb in enumerate(self.dataset.classes())]
    else:
      y = [yb[ix].astype(int) for yb in self.dataset.classes()]

    return X, y, ix