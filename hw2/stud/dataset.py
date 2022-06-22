from torch.utils.data import Dataset
from vocab import Vocabulary
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
import torch
from bertEmbedder import bertEmbeddings
from nltk.tokenize.treebank import TreebankWordDetokenizer


####################################################################
#Dataset class for predicate identification and classification task#
####################################################################

class PICDataset(Dataset):
    def __init__(self, sentences, labels, vocab):
        self.sentences = sentences
        self.labels = labels
        self.vocab = vocab
        self.no_pred_ids = []

        checkpoint = "bert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, clean_text=False)
        model = AutoModel.from_pretrained(checkpoint)
        self.bert_embedder = bertEmbeddings(model=model, tokenizer=tokenizer, device=torch.device('cpu'))

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
            return pred

    def prepare_sentences(self):
        assert self.vocab.constructed, 'Vocabulary should be constructed before preparing the sentences'
        sentences = []
        for sentence_key in self.sentences.keys():
            if self.get_predicates(self.sentences[sentence_key]):
                for i in self.pred.keys():
                    sentence = dict()
                    #sent = TreebankWordDetokenizer().detokenize(self.sentences[sentence_key]['words'])
                    sentence['words'] = self.vocab.words2indeces(self.sentences[sentence_key]['words'])
                    sentence['words_values'] = self.sentences[sentence_key]['words']
                    sentence['lemmas'] = self.vocab.lemmas2indeces(self.sentences[sentence_key]['lemmas'])
                    sentence['predicates'] = [self.vocab.pred2id['_']]*len(self.sentences[sentence_key]['predicates'])
                    sentence['predicates'][i] = self.vocab.pred2id.get(self.pred[i], self.vocab.pred2id['<UNK>'])
                    sentence['pos_tags'] = self.vocab.pts2indices(self.sentences[sentence_key]['pos_tags'])
                    sentence['roles'] = self.vocab.roles2indices(self.labels[sentence_key]['roles'][i])
                    sentence['id'] = sentence_key
                    sentence['bert_embed'] = self.bert_embedder.get_sentence_vector(self.sentences[sentence_key]['words'])
                    sentence['pred_indeces'] = i

                    sentences.append(sentence)
            else:
                self.no_pred_ids.append(sentence_key)

        self.features = sentences

    def collate_fn(self, batch):
        words_batch = [sentence['words'] for sentence in batch]
        predicates_batch = [sentence['predicates'] for sentence in batch]
        pos_tags_batch = [sentence['pos_tags'] for sentence in batch]
        lemmas_batch = [sentence['lemmas'] for sentence in batch]
        roles_batch = [sentence['roles'] for sentence in batch]
        bert_batch = [sentence['bert_embed'] for sentence in batch]
        ids_batch = [sentence['id'] for sentence in batch]
        words_value_batch = [sentence['words_values'] for sentence in batch]
        sentence = dict()
        sentence['words'] = pad_sequence([torch.as_tensor(sample) for sample in words_batch], batch_first=True)
        sentence['predicates'] = pad_sequence([torch.as_tensor(sample) for sample in predicates_batch], batch_first=True)
        sentence['pos_tags'] = pad_sequence([torch.as_tensor(sample) for sample in pos_tags_batch], batch_first=True)
        sentence['lemmas'] = pad_sequence([torch.as_tensor(sample) for sample in lemmas_batch], batch_first=True)
        sentence['roles'] = pad_sequence([torch.as_tensor(sample) for sample in roles_batch], batch_first=True)
        sentence['bert_embed'] = pad_sequence(bert_batch, batch_first=True)
        sentence['id'] = ids_batch
        sentence['words_values'] =words_value_batch

        return sentence


        

