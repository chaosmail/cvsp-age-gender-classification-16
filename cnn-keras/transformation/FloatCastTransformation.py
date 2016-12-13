"""
Author: Christoph Koerner
Student-ID: 0726266
"""
import numpy as np

from .SampleTransformation import *


class FloatCastTransformation(SampleTransformation):
  # Casts the sample datatype to single-precision float (e.g. numpy.float32).

  def apply(self, sample):
    # Apply the transformation and return the transformed version.
    return sample.astype(np.float32)