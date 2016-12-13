"""
Author: Christoph Koerner
Student-ID: 0726266
"""
from .SampleTransformation import *


class ReshapeTransformation(SampleTransformation):
  # Subtract a scalar from all features.

  def __init__(self, value):
    # Constructor.
    # value is a scalar to subtract.
    self._shape = value

  def apply(self, sample):
    # Apply the transformation and return the transformed version.
    # The sample datatype must be single-precision float.
    return sample.reshape(self._shape)

  def value(self):
    # Return the subtracted value.
    return self._shape