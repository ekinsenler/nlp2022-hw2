from torch.utils.data import Dataset
from vocab import Vocabulary

####################################################################
#Dataset class for predicate identification and classification task#
####################################################################

class PICDataset(Dataset):
    def __init__(self, sentences, labels, vocab):
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab

    def __len__(self) -> int:
        assert hasattr(self, 'features'), 'Call prepare features first'
        return len(self.features)

    def __getitem__(self, key: int):
        assert hasattr(self, 'features'), 'Call prepare features first'
        return self.features[key]

    def get_predicates(self, sentence):
        pred = dict()
        if all(elem == sentence['predicates'][0] for elem in sentence['predicates']):
            #all elements are same in predicate list which means that there is no predicate
            return False
        else:
            for i, p in enumerate(sentence['predicates']):
                if p != '_':
                    pred[i] = p
            self.pred = pred
            return True

    def prepare_sentences(self):
        assert self.vocab.constructed, 'Vocabulary should be constructed before preparing the sentences'
        sentences = []
        for sentence_key in self.sentences.keys():
            if self.get_predicates(self.sentences[sentence_key]):
                for i in self.pred.keys():
                    sentence = dict()
                    sentence['words'] = self.vocab.words2indeces(self.sentences[sentence_key]['words'])
                    sentence['lemmas'] = self.vocab.lemmas2indeces(self.sentences[sentence_key]['lemmas'])
                    sentence['predicates'] = [self.vocab.pred2id['_']]*len(self.sentences[sentence_key]['predicates'])
                    sentence['predicates'][i] = self.vocab.pred2id.get(self.pred[i], self.vocab.pred2id['<UNK>'])
                    sentence['pos_tags'] = self.vocab.pts2indices(self.sentences[sentence_key]['pos_tags'])
                    sentence['roles'] = self.vocab.roles2indices(self.labels[sentence_key]['roles'])
                    sentences.append(sentence)
        self.features = sentences

        

