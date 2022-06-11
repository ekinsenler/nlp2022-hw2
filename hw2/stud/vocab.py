import torch

from config import config as cfg
class Vocabulary:
    def __init__(self):
        self.id2lemma = {0:'<PAD>', 1: '<SOS>', 2:'<EOS>', 3:'<UNK>', 4:'<SEP>'}
        self.lemma2id = {k:j for j,k in self.id2lemma.items()}
        self.id2word = {0:'<PAD>', 1: '<SOS>', 2:'<EOS>', 3:'<UNK>', 4:'<SEP>'}
        self.word2id = {k:j for j,k in self.id2word.items()}
        self.id2pred = {0:'<PAD>', 1:'<UNK>'}
        self.pred2id = {k:j for j,k in self.id2pred.items()}
        self.id2pt = {0:'<PAD>'}
        self.pt2id = {k:j for j,k in self.id2pt.items()}
        self.id2role = {0:'<PAD>'}
        self.role2id = {k:j for j,k in self.id2role.items()}

    ####Adjusting the indeces with the special tokens
        self.index_words = 4
        self.index_lemmas = 4
        self.index_predicates = 2
        self.index_pts = 1
        self.index_roles = 1

        self.constructed = False

    def __len__(self):
        return len(self.id2word)

    def construct_vocabulary(self, sentences, labels):
        for sentence, label in zip(sentences.values(), labels.values()):
            for word, lemma, predicate, pt, *role in zip(sentence['words'], sentence['lemmas'], sentence['predicates'], sentence['pos_tags'], *label['roles'].values()):
                if word not in self.id2word.values():
                    self.id2word[self.index_words] = word
                    self.word2id[word] = self.index_words
                    self.index_words += 1
                if lemma not in self.id2lemma.values():
                    self.id2lemma[self.index_lemmas] = lemma
                    self.lemma2id[lemma]= self.index_lemmas
                    self.index_lemmas += 1
                if predicate not in self.id2pred.values():
                    self.id2pred[self.index_predicates] = predicate
                    self.pred2id[predicate] = self.index_predicates
                    self.index_predicates += 1
                if pt not in self.id2pt.values():
                    self.id2pt[self.index_pts] = pt
                    self.pt2id[pt] = self.index_pts
                    self.index_pts += 1
                for i_role in role:
                    if i_role not in self.id2role.values():
                        self.id2role[self.index_roles] = i_role
                        self.role2id[i_role] = self.index_roles
                        self.index_roles += 1
            self.constructed = True
            torch


    def words2indeces(self, words):
        indexed_words = []
        for word in words:
            if word in self.word2id.keys():
                indexed_words.append(self.word2id[word])
            else:
                indexed_words.append(self.word2id['<UNK>'])
        return indexed_words

    def lemmas2indeces(self, lemmas):
        indexed_lemmas = []
        for lemma in lemmas:
            if lemma in self.lemma2id.keys():
                indexed_lemmas.append(self.lemma2id[lemma])
            else:
                indexed_lemmas.append(self.lemma2id['<UNK>'])
        return indexed_lemmas

    def preds2indices(self, preds):
        indexed_preds = []
        for pred in preds:
            if pred in self.pred2id.keys():
                indexed_preds.append(self.pred2id[pred])
            else:
                indexed_preds.append(self.pred2id['<UNK>'])
        return indexed_preds

    def pts2indices(self, pts):
        indexed_pts = []
        for pt in pts:
            if pt in self.pt2id.keys():
                indexed_pts.append(self.pt2id[pt])
            # else:
            #     indexed_pts.append(self.pt2id['<UNK>'])
        return indexed_pts

    def roles2indices(self, roles):
        indexed_roles = []
        for role in roles:
            if role in self.role2id.keys():
                indexed_roles.append(self.role2id[role])
            # else:
            #     indexed_roles.append(self.role2id['<UNK>'])
        return indexed_roles

    def load(self, filepath):
        vocab = torch.load()


