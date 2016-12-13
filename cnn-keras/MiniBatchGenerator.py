"""
Author: Christoph Koerner
Student-ID: 0726266
"""
import numpy as np


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

  def batch(self, bid):
    # Return the bid-th minibatch.
    # Batch IDs start with 0 and are consecutive.
    # Throws an error if the minibatch does not exist.
    if bid >= self.nbatches():
      raise ValueError('Cannot retrieve batch %i from %i batches' % (bid, self.nbatches()))

    batchsize = self.bs
    if self.dataset.size() % self.bs and bid == self.nbatches() - 1:
      batchsize = self.dataset.size() % self.bs

    start = self.bs * bid
    
    X = np.zeros((batchsize, *self.tform.apply(self.dataset.sample(0)).shape))    
    y = np.zeros((batchsize), dtype=int)
    ids = np.zeros((batchsize), dtype=int)

    for i in range(batchsize):
      _id = self.indices[start + i]
      if self.tform is not None:
        X[i] = self.tform.apply(self.dataset.sample(_id))
      else:
        X[i] = self.dataset.sample(_id)
      y[i] = self.dataset.sample_class(_id)
      ids[i] = _id

    return X, y, ids