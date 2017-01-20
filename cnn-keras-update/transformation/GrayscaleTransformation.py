"""
Author: Christoph Koerner
Student-ID: 0726266
"""
from .SampleTransformation import *


class GrayscaleTransformation(SampleTransformation):
  # Subtract a scalar from all features.

  def apply(self, sample):
    # Apply the transformation and return the transformed version.
    # The sample datatype must be single-precision float.
    r, g, b = sample[:,:,0], sample[:,:,1], sample[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b
    