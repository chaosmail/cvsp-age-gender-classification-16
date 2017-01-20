"""
Author: Christoph Koerner
Student-ID: 0726266
"""
import numpy as np


class ImageVectorizer():
  """Wraps an image dataset and exposes its contents.
  Samples obtained using sample() are returned as 1D feature vectors.
  Use devectorize() to convert a vector back to an image."""

  def __init__(self, dataset):
    """Ctor. dataset is the dataset to wrap (type ImageDataset)."""
    self.dataset = dataset

  def devectorize(self, fvec):
    """Convert a feature vector fvec obtained using sample()
    back to an image and return the converted version."""

    # Helper function to convert image col into
    # RGB matrix with the dimensions (SIZE, SIZE, 3)
    dim = self.dataset.dim
    to_rgb = lambda data: np.dstack((
      # extract R channel -> reshape to 32x32
      data[:dim[0]*dim[1]].reshape(dim),
      
      # extract G channel -> reshape to 32x32
      data[dim[0]*dim[1]:dim[0]*dim[1]+dim[0]*dim[1]].reshape(dim),

      # extract B channel -> reshape to 32x32
      data[dim[0]*dim[1]+dim[0]*dim[1]:].reshape(dim)
    ))

    return to_rgb(fvec)

  def size(self):
    # Return the size of the dataset (number of samples).
    return self.dataset.size()

  def nclasses(self):
    # Return the number of different classes.
    # Class labels start with 0 and are consecutive.
    return self.dataset.nclasses()

  def classname(self, cid):
    # Return the name of a class as a string.
    return self.dataset.classname(cid)

  def sample(self, sid):
    # Return the sid-th sample in the dataset, and the
    # corresponding class label. Depending of your language,
    # this can be a Matlab struct, Python tuple or dict, etc.
    # Sample IDs start with 0 and are consecutive.
    # Throws an error if the sample does not exist.
    return self.dataset.sample(sid).reshape(-1)

  def sample_classname(self, sid):
    """Returns the classname of a sample"""
    return self.dataset.sample_classname(sid)

  def sample_class(self, sid):
    """Returns the class of a sample"""
    return self.dataset.sample_class(sid)