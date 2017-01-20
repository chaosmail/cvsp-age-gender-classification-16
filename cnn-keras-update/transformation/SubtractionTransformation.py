"""
Author: Christoph Koerner
Student-ID: 0726266
"""
from .SampleTransformation import *
from .TransformationSequence import *

class SubtractionTransformation(SampleTransformation):
  # Subtract a scalar from all features.

  @staticmethod
  def from_dataset_mean(dataset, tform=None):
    # Return a transformation that will subtract by the global mean
    # over all samples and features in a dataset.
    # tform is an optional SampleTransformation to apply before computation.
    t = TransformationSequence()

    # Add the additional transformation tform
    if tform is not None:
      t.add_transformation(tform)

    # Add the mean subtraction transform
    t.add_transformation(SubtractionTransformation(dataset.get_mean()))

    return t

  def __init__(self, value):
    # Constructor.
    # value is a scalar to subtract.
    self._value = value

  def apply(self, sample):
    # Apply the transformation and return the transformed version.
    # The sample datatype must be single-precision float.
    return sample - self._value

  def value(self):
    # Return the subtracted value.
    return self._value