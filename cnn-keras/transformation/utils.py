from .FloatCastTransformation import *
from .ResizeTransformation import *
from .TransformationSequence import *
from .PerChannelSubtractionImageTransformation import *
from .PerChannelDivisionImageTransformation import *
from .ReshapeTransformation import *
from .GrayscaleTransformation import *


def get_normalization_transform(means=None, stds=None, resize_to=None, reshape_to=None, grayscale=False):
  print(" Initializing Transformations")
  tform = TransformationSequence()

  if resize_to is not None:
    t = ResizeTransformation(resize_to)
    tform.add_transformation(t)
    print("  %s" % t.__class__.__name__)

  t = FloatCastTransformation()
  tform.add_transformation(t)
  print("  %s" % t.__class__.__name__)

  if means is not None:
    t = PerChannelSubtractionImageTransformation(np.array(means, dtype=np.float32))
    tform.add_transformation(t)
    print("  %s (%s)" % (t.__class__.__name__, str(t.values())))

  if stds is not None:
    t = PerChannelDivisionImageTransformation(np.array(stds, dtype=np.float32))
    tform.add_transformation(t)
    print("  %s (%s)" % (t.__class__.__name__, str(t.values())))

  if grayscale is True:
    t = GrayscaleTransformation()
    tform.add_transformation(t)
    print("  %s" % (t.__class__.__name__))

  if reshape_to is not None:
    t = ReshapeTransformation(reshape_to)
    tform.add_transformation(t)
    print("  %s %s" % (t.__class__.__name__, str(t.value())))

  return tform