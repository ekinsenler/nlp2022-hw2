class Vocabulary:
    def __init__(self):
        self.id2word = {0:'<PAD>', 1: '<SOS>', 2:'<EOS>', 3:'<UNK>', 4:'<SEP>'}
        self.word2id = {k:j for j,k in self.id2word.items()}
        self.id2pred = {0:'<PAD>', 1:'<UNK>'}
        self.pred2id = {k:j for j,k in self.id2pred.items()}
        self.id2role = {0:'<PAD>'}
        self.role2id = {k:j for j,k in self.id2role.items()}


    # We already added 4 element to the vocab
        self.index_lemmas = 4
        self.index_predicates = 2
        self.index_roles = 1
    def __len__(self):
        return len(self.id2word)
    def build_vocab(self, sentences, labels):
        for sentence, label in zip(sentences.values(), labels.values()):
            for lemma, predicate, role in zip(sentence['lemmas'], sentence['predicates'], label['roles']):
                if lemma not in self.id2word.items():
                    self.id2word[self.index_lemmas] = lemma
                    self.word2id[lemma]= self.index_lemmas
                    self.index_lemmas += 1
                if predicate not in self.id2pred.items():
                    self.id2pred[self.index_predicates] = predicate
                    self.pred2id[predicate] = self.index_predicates
                    self.index_predicates += 1
                if role not in self.id2role.items():
                    self.id2role[self.index_roles] = role
                    self.role2id[role] = self.index_predicates
                    self.index_roles += 1


    def sen2index(self, sen):
        indexed_sen = []
        for word in sen:
            if word in self.word2id.keys():
                indexed_sen.append(self.word2id[word])
            else:
                indexed_sen.append(self.word2id['<UNK>'])
        return indexed_sen


