"""
Author: Christoph Koerner
Student-ID: 0726266
"""
from .SampleTransformation import *


class IdentityTransformation(SampleTransformation):
  # A transformation that does not do anything.

  def apply(self, sample):
    # Apply the transformation and return the transformed version.
    return sample