from .ImdbWikiDataset import *


class ImdbWikiGenderDataset(ImdbWikiDataset):

  def __init__(self, fdir, split):
    ImdbWikiDataset.__init__(self, fdir, split)

    self.labels = self.load_labels('gender')
    self.label_names = ['female', 'male']