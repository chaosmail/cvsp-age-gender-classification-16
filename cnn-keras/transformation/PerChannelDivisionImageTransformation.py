"""
Author: Christoph Koerner
Student-ID: 0726266
"""
from .SampleTransformation import *
from .TransformationSequence import *


class PerChannelDivisionImageTransformation(SampleTransformation):
  # Perform per-channel division of of image samples with a scalar.

  @staticmethod
  def from_dataset_stddev(dataset, tform=None):
    # Return a transformation that will divide by the global standard deviation
    # over all samples and features in a dataset, independently for every color channel.
    # tform is an optional SampleTransformation to apply before computation.
    # samples must be 3D tensors with shape [rows,cols,channels].
    # rows, cols, channels can be arbitrary values > 0.
    t = TransformationSequence()

    # Add the additional transformation tform
    if tform is not None:
      t.add_transformation(tform)

    # Add the mean subtraction transform
    t.add_transformation(PerChannelDivisionImageTransformation(dataset.get_stddev(per_channel=True)))

    return t

  def __init__(self, values):
    # Constructor.
    # values is a vector of c divisors, one per channel.
    # c can be any value > 0.
    self._values = values

  def apply(self, sample):
    # Apply the transformation and return the transformed version.
    # sample must be a 3D tensor with shape [rows,cols,c].
    # The sample datatype must be single-precision float.

    sample[:,0] /= self._values[0]
    sample[:,1] /= self._values[1]
    sample[:,2] /= self._values[2]
    
    return sample

  def values(self):
    # Return the divisors.
    return self._values