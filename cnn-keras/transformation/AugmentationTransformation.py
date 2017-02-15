"""
Author: Christoph Koerner
Student-ID: 0726266
"""
from .SampleTransformation import *
from keras.preprocessing import image
import numpy as np


class AugmentationTransformation(SampleTransformation):
  # Perform horizontal mirroring of samples with a given probability.

  def __init__(self, shiftX=0.1, shiftY=0.1, rot=0, shear=0.01, zoomMin=0.95, zoomMax=1.05, flipH=True, flipV=False):
    # Constructor.
    self.shiftX = shiftX
    self.shiftY = shiftY
    self.rot = rot
    self.shear = shear
    self.zoomMin = zoomMin
    self.zoomMax = zoomMax
    self.flipH = flipH
    self.flipV = flipV

  def apply(self, samples):
    # Apply the transformation and return the transformed version.
    # sample must be a 3D tensor with shape [rows,cols,channels].
    out = np.zeros(samples.shape)

    for i in range(len(samples)):
      out[i] = samples[i]

      if self.shiftX > 0 or self.shiftY > 0:
        out[i] = image.random_shift(out[i], self.shiftX, self.shiftY)
  
      if self.rot > 0:
        out[i] = image.random_rotation(out[i], self.rot)
  
      if self.shear > 0:
        out[i] = image.random_shear(out[i], self.shear)
      
      if self.zoomMin > 0 and self.zoomMax > 0:
        out[i] = image.random_zoom(out[i], [self.zoomMin, self.zoomMax])

      if self.flipH and np.random.uniform() < 0.5:
        out[i] = image.flip_axis(out[i], 1)

      if self.flipV and np.random.uniform() < 0.5:
        out[i] = image.flip_axis(out[i], 2)

    return out