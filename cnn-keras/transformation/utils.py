from .FloatCastTransformation import *
from .ResizeTransformation import *
from .MultiplyTransformation import *
from .DivisionTransformation import *
from .TransformationSequence import *
from .TransposeTransformation import *
from .PerChannelSubtractionImageTransformation import *
from .PerChannelDivisionImageTransformation import *
from .ReshapeTransformation import *
from .GrayscaleTransformation import *
from .AugmentationTransformation import *


def get_normalization_transform(means=None, stds=None, resize_to=None, transpose_to=None,
    reshape_to=None, grayscale=False, scale_to=None, normalize=False, augmentation=False,
    verbose=True):
  
  if verbose:
    print(" Initializing Transformations")
  tform = TransformationSequence()

  if resize_to is not None:
    t = ResizeTransformation(resize_to)
    tform.add_transformation(t)
    if verbose:
      print("  %s" % type(t).__name__)

  t = FloatCastTransformation()
  tform.add_transformation(t)
  if verbose:
    print("  %s" % type(t).__name__)

  if means is not None:
    t = PerChannelSubtractionImageTransformation(np.array(means, dtype=np.float32))
    tform.add_transformation(t)
    if verbose:
      print("  %s (%s)" % (type(t).__name__, str(t.values())))

  if stds is not None:
    t = PerChannelDivisionImageTransformation(np.array(stds, dtype=np.float32))
    tform.add_transformation(t)
    if verbose:
      print("  %s (%s)" % (type(t).__name__, str(t.values())))

  if grayscale is True:
    t = GrayscaleTransformation()
    tform.add_transformation(t)
    if verbose:
      print("  %s" % (type(t).__name__))

  if scale_to is not None:
    t = MultiplyTransformation(scale_to)
    tform.add_transformation(t)
    if verbose:
      print("  %s %s" % (type(t).__name__, str(t.value())))    

  if transpose_to is not None:
    t = TransposeTransformation(transpose_to)
    tform.add_transformation(t)
    if verbose:
      print("  %s %s" % (type(t).__name__, str(t.value())))

  if reshape_to is not None:
    t = ReshapeTransformation(reshape_to)
    tform.add_transformation(t)
    if verbose:
      print("  %s %s" % (type(t).__name__, str(t.value())))

  if normalize:
    t = DivisionTransformation(255)
    tform.add_transformation(t)
    if verbose:
      print("  %s %s" % (type(t).__name__, str(t.value())))

  if augmentation:
    t = AugmentationTransformation()
    tform.add_transformation(t)
    if verbose:
      print("  %s" % (type(t).__name__))

  return tform