from .ImdbWikiDataset import *


class ImdbWikiAgeDataset(ImdbWikiDataset):

  def __init__(self, fdir, split):
    ImdbWikiDataset.__init__(self, fdir, split)

    self.labels = self.load_labels('age')
    self.label_names = [
      '[0 - 15]', '[16 - 20]', '[21 - 25]', '[26 - 30]', '[31 - 35]',
      '[36 - 40]', '[41 - 45]', '[46 - 50]', '[51 - 55]', '[56 - 100]'
    ]