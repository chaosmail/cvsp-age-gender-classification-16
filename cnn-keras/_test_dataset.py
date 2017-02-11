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

def flip(X):
  return X.transpose((2,1,0))

def print_summary(dataset_name, sets, sid=110):
  print("\nSummary of %s", dataset_name)

  for s in sets:
    print("[%s] %i classes, name of class #1: %s" 
      % (s.split, s.nclasses(), s.classname(1)))

  for s in sets:
    print("\n[%s] %i samples" % (s.split, s.size()))
    for lid in range(s.nclasses()):
      print(" Class #%i: %i samples" %
        (lid, s.samples()[s.classes() == lid].shape[0]))

  print()
  for s in sets:
    print("[%s] Sample #%i: %s" %
      (s.split, sid, s.sample_classname(sid)))

  for s in sets:
    data = s.sample(sid) * 255
    img = Image.fromarray(flip(data).astype(np.uint8))
    img.save('%s_%s_sample#%i.png' % (dataset_name, s.split, sid))


# # the dataset is in the directory
# DATASET_DIR = '../data/imdb-wiki-dataset'

# from dataset import ImdbWikiAgeDataset as Dataset

# # Print a summary of the dataset
# print_summary('ImdbWikiAgeDataset', [
#   Dataset(DATASET_DIR, 'train'),
#   Dataset(DATASET_DIR, 'val'),
#   Dataset(DATASET_DIR, 'test')
# ])


# from dataset import ImdbWikiGenderDataset as Dataset

# # Print a summary of the dataset
# print_summary('ImdbWikiGenderDataset', [
#   Dataset(DATASET_DIR, 'train'),
#   Dataset(DATASET_DIR, 'val'),
#   Dataset(DATASET_DIR, 'test')
# ])

# print('\n')

# the dataset is in the directory
DATASET_DIR = '/data/imdb-wiki-dataset'

from dataset import ImdbWikiAgeDataset as Dataset

print_summary('ImdbWikiAgeDataset', [
  Dataset(DATASET_DIR, 'train')
])

print_summary('ImdbWikiAgeDataset', [
  Dataset(DATASET_DIR, 'val')
])

print_summary('ImdbWikiAgeDataset', [
  Dataset(DATASET_DIR, 'test')
])