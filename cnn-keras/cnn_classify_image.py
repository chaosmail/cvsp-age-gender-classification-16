import argparse
import numpy as np
from PIL import Image
from keras.models import load_model

from transformation import *

cifar10_classnames = {
  0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
  5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}


def load_image(image_path):
  print("Loading image ...")
  try:
    with Image.open(image_path) as img:
      img_data = np.asarray(img)
      print(" Shape: %s" % str(img_data.shape))
  except FileNotFoundError as e:
    print('Failed loading image %s' % image_path)
    exit()
  return img_data


def load_classifier(model_path):
  print("Loading classifier ...")
  try:
    model = load_model(model_path)
    print(" Input shape: %s, %i classes" % (
      str(model.layers[0].batch_input_shape[1:]), model.layers[-1].output_dim))
  except OSError as e:
    print('Failed loading model %s' % model_path)
    exit()
  return model


def preprocess_image(sample, means=None, stds=None, dims=None, out_shape=None):
  print("Preprocessing image ...")
  print(" Transformations in order:")
  tform = TransformationSequence()

  if dims is not None:
    t = ResizeTransformation(dims)
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

  if out_shape is not None:
    t = ReshapeTransformation(out_shape)
    tform.add_transformation(t)
    print("  %s %s" % (t.__class__.__name__, str(t.value())))

  sample = tform.apply(sample)
  print(" Result: shape: %s, dtype: %s, mean: %.3f, std: %.3f" % (
    sample.shape, sample.dtype, sample.mean(), sample.std()))

  return sample


def classify_image(model, sample):
  print("Classifying image ...")

  # Expand the sample (3,32,32) to batch dimension (1,3,32,32)
  X = np.expand_dims(sample, axis=0)
  
  # Evaluate the model and compute the class probabilities
  # Reshape the result (1, 10) to a vector (10, )
  probas = model.predict_proba(X, batch_size=1, verbose=0).reshape(-1)
  
  # Get the class index of the highest probability
  class_id = np.argmax(probas)
  class_acc = probas[class_id]
  class_label = cifar10_classnames[class_id]

  print(" Class scores: %s" % (str(probas)))
  print(" ID of most likely class: %i (score: %.2f)" % (class_id, class_acc))
  print(" Label of most likely class: %s" % (class_label))

  return probas


def perform_classification(model_path, image_path, means=None, stds=None):
  image = load_image(image_path)
  model = load_classifier(model_path)
  sample = preprocess_image(image, means=means, stds=stds, dims=(32,32,3), out_shape=(3,32,32))
  probas = classify_image(model, sample)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', dest='model',
    help='Path to the model', default='model_best.h5')
  parser.add_argument('--image', dest='image', 
    help='Path to the RGB input image')
  parser.add_argument('--means', dest='means', type=float, nargs='+',
    help='Mean subtraction for preprocessing')
  parser.add_argument('--stds', dest='stds', type=float, nargs='+',
    help='Stddev division for preprocessing')

  print("Parsing arguments ...")
  args = parser.parse_args()

  print(" Model: %s" % args.model)
  print(" Image: %s" % args.image)
  print(" Means: %s" % str(args.means))
  print(" Stds: %s" % str(args.stds))

  perform_classification(args.model, args.image, args.means, args.stds)

if __name__ == '__main__':
  main()