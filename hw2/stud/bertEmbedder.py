from transformers import AutoModel, AutoTokenizer
import torch

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

class bertEmbeddings:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer, device:torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)

    def _get_word_2_indices(self, encoded):

        # we have to collect all the subtokenizations of the words
        w_id2input_ids = dict()
        # we have to keep track of the indices that compose the word
        # so we can use them to extract the embeddings
        w_id2_indices = dict()
        for idx, (w_id, input_id) in enumerate(zip(encoded.word_ids(), encoded.input_ids[0])):
            w_id2input_ids.setdefault(w_id, []).append(input_id)
            w_id2_indices.setdefault(w_id, []).append(idx)

        # now let's decode all the words that we have
        word2_indices = dict()
        for w_id, input_ids in w_id2input_ids.items():
            word = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            word = word.replace(" ", "")
            word2_indices[word] = w_id2_indices[w_id]
        return word2_indices

    # produce word vectors by averaging
    # all embeddings of the bpes of the word
    def get_word_vector(self, sent, word):
        # encode_plus returns additional information such as the word ids.
        encoded = self.tokenizer.encode_plus(sent, return_tensors="pt")
        # to guarantee the tensors are on the same device as the model
        encoded = encoded.to(self.model.device)
        # get the mapping between the tokens and the indices of the embeddings
        word2_indices = self._get_word_2_indices(encoded)
        with torch.no_grad():
            output = self.model(**encoded)
        # we get the word embeddings of the last layer of the last hidden state
        last_hidden_state = output[0]
        # bert computes embeddings in batches we gave a single sentence
        # so we have just one element in the batch and can extract the token embeddings
        sentence_token_embeddings = last_hidden_state[0]

        # we take the embeddings correspondings to the indices of the BPEs that compose our word
        word_bpes_embeddings = sentence_token_embeddings[word2_indices[word]]
        # word_bpes_embeddings is a 2D-shaped matrix. dim=0 is needed to specify the axis to compute the mean along.
        # In this case, dim=0 is chosen since we want to average all the row vectors, which correspond to bpe embeddings.
        return word_bpes_embeddings.mean(dim=0)

    def get_sentence_vector(self, sent_token):
        # encode_plus returns additional information such as the word ids.
        encoded = self.tokenizer.encode_plus(sent_token, return_tensors="pt", is_split_into_words=True)
        # to guarantee the tensors are on the same device as the model
        encoded = encoded.to(self.model.device)
        # get the mapping between the tokens and the indices of the embeddings
        word2_indices = self._get_word_2_indices(encoded)
        with torch.no_grad():
            output = self.model(**encoded)
        # we get the word embeddings of the last layer of the last hidden state
        last_hidden_state = output[0]
        # bert computes embeddings in batches we gave a single sentence
        # so we have just one element in the batch and can extract the token embeddings
        sentence_token_embeddings = last_hidden_state[0]
        word_bpes_embeddings = []
        for word in sent_token:
            if word != '[SEP]' and word != '[CLS]' and word in word2_indices:
                word_bpes_embeddings.append(sentence_token_embeddings[word2_indices[word]].mean(dim=0))
            elif word2_indices.get('[UNK]', 0):
                word_bpes_embeddings.append(sentence_token_embeddings[word2_indices['[UNK]']].mean(dim=0))
            else:
                for w in word2_indices.keys():
                    if w == '[CLS][SEP]':
                        continue
                    if all([char in word for char in w]):
                        word_bpes_embeddings.append(sentence_token_embeddings[word2_indices[w]].mean(dim=0))
        return torch.stack(word_bpes_embeddings)

# sample_sentence_token = ['World', 'of', 'warcraft', 'is', 'a', 'drug']
# sample_sentence = 'World of warcraft is a drug'
# sample_word = 'drug'
# embeder = bertEmbeddings(model,tokenizer, torch.device('cpu'))
# sample_word_embeddings = embeder.get_word_vector(sample_sentence, sample_word)
# sample_sentence_embeddings = embeder.get_sentence_vector(sample_sentence, sample_sentence_token)

print('Done...')
