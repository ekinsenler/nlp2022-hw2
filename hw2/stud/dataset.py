from torch.utils.data import Dataset
from vocab import Vocabulary

####################################################################
#Dataset class for predicate identification and classification task#
####################################################################

class PICDataset(Dataset):
    def __init__(self, sentences, labels):
        self.vocab = Vocabulary()
        self.sentences = sentences
        self.labels = labels
        self.vocab.build_vocab(self.sentences, self.labels)
        self.data_points = None


    def __len__(self) -> int:
        return len(self.data_points)
    def __getitem__(self, key: int):
        return self.data_points[key]

    def get_predicates(self, sentence):
        pred = dict()
        if all(elem == sentence['predicates'][0] for elem in sentence['predicates']):
            #all elements are same in predicate list which means there is no predicate
            return False
        else:
            for i, p in enumerate(sentence['predicates']):
                if p != '_':
                    pred[i] = p
            self.pred = pred


    def prepare_data_points(self):
        for sentence_key in self.sentences.keys():
            if self.get_predicates(self.sentences[sentence_key]):
                pass

        

