"""
Author: Christoph Koerner
Student-ID: 0726266

Recap: Test with the corrected abstract classes
"""
import fs
import numpy as np
from PIL import Image

# Helper function to return the length of a filter
len_filter = lambda f, l: len(list(filter(f, l)))

def print_summary(dataset_name, sets):
  print("Summary of %s", dataset_name)

  for s in sets:
    print("[%s] %i classes, name of class #1: %s" 
      % (s.split, s.nclasses(), s.classname(1)))

  for s in sets:
    print("\n[%s] %i samples" % (s.split, s.size()))
    for lid in range(s.nclasses()):
      print(" Class #%i: %i samples" %
        (lid, s.samples()[s.classes() == lid].shape[0]))

  sid = 499

  print()
  for s in sets:
    print("[%s] Sample #%i: %s" %
      (s.split, sid, s.sample_classname(sid)))

  for s in sets:
    data = s.sample(sid) * 255
    img = Image.fromarray(data.reshape((48,48,3)).astype(np.uint8))
    img.save('%s_%s_sample#%i.png' % (dataset_name, s.split, sid))


# the dataset is in the directory
DATASET_DIR = '../data/packaged'

from dataset import TinyImdbWikiAgeDataset as Dataset

# Print a summary of the dataset
print_summary('TinyImdbWikiAgeDataset', [
  Dataset(DATASET_DIR, 'train'),
  Dataset(DATASET_DIR, 'val'),
  Dataset(DATASET_DIR, 'test')
])


from dataset import TinyImdbWikiGenderDataset as Dataset

# Print a summary of the dataset
print_summary('TinyImdbWikiGenderDataset', [
  Dataset(DATASET_DIR, 'train'),
  Dataset(DATASET_DIR, 'val'),
  Dataset(DATASET_DIR, 'test')
])