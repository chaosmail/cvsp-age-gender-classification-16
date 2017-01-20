"""
Author: Christoph Koerner
Student-ID: 0726266
"""
from .SampleTransformation import *
from scipy.ndimage.interpolation import zoom
import numpy as np


class ResizeTransformation(SampleTransformation):
  # Subtract a scalar from all features.

  def __init__(self, value):
    # Constructor.
    # value is a scalar to subtract.
    self._shape = value

  def apply(self, sample):
    # Apply the transformation and return the transformed version.
    # The sample datatype must be single-precision float.
    return zoom(sample, tuple(np.array(self._shape) / np.array(sample.shape)))

  def value(self):
    # Return the subtracted value.
    return self._shape