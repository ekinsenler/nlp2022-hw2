import torch
from torch.nn import Module, LSTM, Linear, Dropout, CrossEntropyLoss, init
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from config import config as cfg
import gensim.downloader as api
from utils import build_pretrain_embedding, load_torch_embedding_layer
from vocab import Vocabulary
from tqdm import trange
import datetime

class SRLModel(Module):
    def __init__(self,  vocab: Vocabulary, device: torch.device, pretrained_embed = cfg['pretrained_embed']):
        super(SRLModel, self).__init__()
        self.input_dim = cfg['word_embed_dim']
        self.num_layers = cfg['num_layers']
        self.hidden_size = cfg['hidden_size']
        self.vocab = vocab
        self.device = device
        weights = api.load(pretrained_embed)
        trimmed_embed = build_pretrain_embedding(weights, self.vocab, embedd_dim = cfg['word_embed_dim'])
        self.word_embeddings = load_torch_embedding_layer(trimmed_embed,padding_idx=0, freeze=True)
        self.lemma_embeddings = torch.nn.Embedding(num_embeddings=vocab.index_lemmas, embedding_dim=cfg['lemma_embed_dim'])
        self.pos_embeddings = torch.nn.Embedding(num_embeddings=vocab.index_pts, embedding_dim=cfg['pos_embed_dim'])
        self.pred_embedding = torch.nn.Embedding(num_embeddings=vocab.index_predicates, embedding_dim=cfg['pred_embed_dim'])
        self.bilstm_input_size = cfg['word_embed_dim'] + cfg['lemma_embed_dim'] + cfg['pos_embed_dim'] + cfg['pred_embed_dim'] + 200 #200 is bert_bilstm hidden size
        self.bert_bilstm = LSTM(input_size= cfg['bert_embed_dim'], hidden_size= 200 // 2, bidirectional=True,
                           dropout=cfg['dropout'], num_layers=self.num_layers, batch_first=True)
        self.bilstm = LSTM(input_size=self.bilstm_input_size, hidden_size= self.hidden_size // 2, bidirectional=True,
                           dropout=cfg['dropout'], num_layers=self.num_layers, batch_first=True)
        self.bilstm_input_size += 200
        self.dropout = Dropout(p=0.2)
        self.hidden2label = Linear(in_features=self.hidden_size, out_features=vocab.index_roles)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg['lr'])
        self.hidden = self.init_hidden(cfg['batch_size'], xavier=True)

    def forward(self, x):
        word_embed = self.word_embeddings(x['words'])
        pos_embed = self.pos_embeddings(x['pos_tags'])
        lemma_embed = self.lemma_embeddings(x['lemmas'])
        pred_embed = self.pred_embedding(x['predicates'])
        bert_embed, _ = self.bert_bilstm(x['bert_embed'])

        packed_embed = torch.cat((word_embed, pos_embed, lemma_embed, pred_embed, bert_embed), dim=2)
        lstm_output_pack, self.hidden = self.bilstm(packed_embed)
        #out = self.dropout(lstm_output_pack)
        out = self.hidden2label(lstm_output_pack)

        return out

    def init_hidden(self, batch_size: int, xavier: bool = True):
        """
        Initialize hidden.
        Args:
            x: (torch.Tensor): input tensor
            hidden_size: (int):
            num_dir: (int): number of directions in LSTM
            xavier: (bool): wether or not use xavier initialization
        """
        if xavier:
            return (init.xavier_normal_(torch.zeros(batch_size, self.num_layers, self.hidden_size // 2).to(self.device)),
                    init.xavier_normal_(torch.zeros(batch_size, self.num_layers, self.hidden_size // 2).to(self.device)))
        #return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)
        return (torch.zeros(batch_size, self.num_layers, self.hidden_size // 2),
                torch.zeros(batch_size, self.num_layers, self.hidden_size // 2))

    def loss_fn(self, outputs, labels):
        batch_size = outputs.size(0)
        max_seq_len = outputs.size(1)
        loss_function = CrossEntropyLoss(ignore_index=0)
        outputs = outputs.contiguous().view(batch_size * max_seq_len, -1)
        total_loss = loss_function(outputs, labels.contiguous().view(batch_size * max_seq_len))

        return total_loss

    def train_net(self, train_data_loader, val_data_loader, train_path, log_dir_path):
        self.writer = SummaryWriter(log_dir_path)
        epoch = 0
        print(f"Starting training on {self.device}, with batch size ({cfg['batch_size']})")
        for ep in trange(cfg['num_epoch']):
            ##############
            ###TRAINING###
            ##############
            batch_number = 0
            self.train()
            train_losses = []
            for x in train_data_loader:
                batch_number += 1
                self.zero_grad()
                self.hidden = self.init_hidden(batch_size=cfg['batch_size'], xavier=True)
                predicts = self(x)
                #predicts = predicts.view(-1, predicts.shape[-1])
                #predicts = torch.argmax(predicts, -1)
                #y = x['roles'].view(-1)
                loss = self.loss_fn(predicts, x['roles'])
                train_losses.append(loss)
                self.writer.add_scalar('train_loss', loss, batch_number)
                loss.backward()
                clip_grad_norm_(self.parameters(), 5.0)
                self.optimizer.step()
            mean_train_loss = sum(train_losses) / len(train_losses)
            self.writer.add_scalar('mean_train_loss', mean_train_loss, ep)
            ################
            ###EVALUATION###
            ################
            valid_losses = []
            self.eval()
            for x in val_data_loader:
                batch_number += 1
                with torch.no_grad():
                    predicts = self(x)
                    # predicts = predicts.view(-1, output.shape[-1])
                    # y = y.view(-1)
                loss = self.loss_fn(predicts, x['roles'])
                valid_losses.append(loss)
                self.writer.add_scalar('valid_loss', loss, batch_number)
            mean_valid_loss = sum(valid_losses) / len(valid_losses)
            self.writer.add_scalar('mean_valid_loss', mean_valid_loss, ep)

            print('epoch: ' + str(ep) + ' # ' + ' loss: ' + str(mean_train_loss.item()) + ' # ' + '  val loss: ' + str(mean_valid_loss.item())  + ' # '+ '\n')
            epoch += 1
        model_name = 'model_' + str(cfg['num_epoch']) + 'epoch_' + str(datetime.datetime.now().timestamp()) + '.pt'
        torch.save(self.state_dict(), train_path / model_name)
        self.writer.close()

        return model_name

