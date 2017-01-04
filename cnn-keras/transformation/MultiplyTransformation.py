"""
Author: Christoph Koerner
Student-ID: 0726266
"""
from .SampleTransformation import *


class MultiplyTransformation(SampleTransformation):
  # Divide all features by a scalar.

  def __init__(self, value):
    # Constructor.
    # value is a scalar divisor != 0.
    if value == 0:
        raise ValueError('Divisor cannot be 0.')
    self._value = value

  def apply(self, sample):
    # Apply the transformation and return the transformed version.
    # The sample datatype must be single-precision float.
    return sample * self._value;

  def value(self):
    # Return the divisor.
    return self._value