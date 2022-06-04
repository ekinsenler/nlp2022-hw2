class Vocabulary:
    def __init__(self):
        self.id2word = {0:'<PAD>', 1: '<SOS>', 2:'<EOS>', 3:'<UNK>', 4:'<SEP>'}
        self.word2id = {k:j for j,k in self.id2word.items()}
        self.id2label = {0:'<PAD>', 1:'<UNK>'}
        self.label2id = {k:j for j,k in self.id2label.items()}

        # We already added 4 element to the vocab
        self.index_lemmas = 4
        self.index_predicates = 2
    def __len__(self):
        return len(self.id2word)
    def build_vocab(self,data):
        for sentences, labels in zip(data[0].values(), data[1].values()):
            pass
        # for lemmas,predicates in zip(data['lemmas'], data['predicates']):
        #     for lemma,predicate in zip(lemmas,predicates):
        #         if lemma not in self.id2word.items():
        #             self.id2word[self.index_lemmas] = lemma
        #             self.word2id[lemma]= self.index_lemmas
        #             self.index_lemmas += 1
        #         if predicate not in self.id2label.items():
        #             self.id2label[self.index_predicates] = predicate
        #             self.label2id[predicate] = self.index_predicates
        #             self.index_predicates += 1

    def sen2index(self, sen):
        indexed_sen = []
        for word in sen:
            if word in self.word2id.keys():
                indexed_sen.append(self.word2id[word])
            else:
                indexed_sen.append(self.word2id['<UNK>'])
        return indexed_sen


