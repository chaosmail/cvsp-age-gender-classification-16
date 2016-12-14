from .TinyImdbWikiDataset import *


class TinyImdbWikiGenderDataset(TinyImdbWikiDataset):

  def __init__(self, fdir, split):
    TinyImdbWikiDataset.__init__(self, fdir, split)

    self.labels = self.load_labels('gender')
    self.label_names = ['female', 'male']