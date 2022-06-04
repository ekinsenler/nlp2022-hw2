from torch.utils.data import Dataset
from vocab import Vocabulary

####################################################################
#Dataset class for predicate identification and classification task#
####################################################################

class PICDataset(Dataset):
    def __init__(self, data):
        self.data = data[0]
        self.vocab = Vocabulary
        self.vocab.build_vocab(self.data)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, key):
        lemmas = self.data[key]['lemmas']
        predicates = self.data[key['predicates']]

    def prepare_samples(self):
        pass

        

