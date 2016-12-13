"""
Author: Christoph Koerner
Student-ID: 0726266
"""
from .SampleTransformation import *
from .TransformationSequence import *


class DivisionTransformation(SampleTransformation):
  # Divide all features by a scalar.

  @staticmethod
  def from_dataset_stddev(dataset, tform=None):
    # Return a transformation that will divide by the global standard deviation
    # over all samples and features in a dataset.
    # tform is an optional SampleTransformation to apply before computation.
    t = TransformationSequence()

    # Add the additional transformation tform
    if tform is not None:
      t.add_transformation(tform)

    # Add the mean subtraction transform
    t.add_transformation(DivisionTransformation(dataset.get_stddev()))

    return t

  def __init__(self, value):
    # Constructor.
    # value is a scalar divisor != 0.
    if value == 0:
        raise ValueError('Divisor cannot be 0.')
    self._value = value

  def apply(self, sample):
    # Apply the transformation and return the transformed version.
    # The sample datatype must be single-precision float.
    return sample / self._value;

  def value(self):
    # Return the divisor.
    return self._value