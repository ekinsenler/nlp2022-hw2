import torch

from hw2.utils import read_dataset
import pathlib
from vocab import Vocabulary
from dataset import PICDataset
from torch.utils.data import DataLoader
from config import config as cfg
from implementation import build_model_34
curr_dir = pathlib.Path(__file__)
proj_dir = curr_dir.parent.parent.parent
hw1_dir = curr_dir.parent.parent

data_train_path= proj_dir/'data'/'data_hw2'/'EN'/'train.json'
data_dev_path=proj_dir/'data'/'data_hw2'/'EN'/'dev.json'
model_path = proj_dir/'model'

def main():
    sentences, labels =read_dataset(data_train_path)
    if (model_path / 'vocab.pt').is_file():
        vocab = torch.load(model_path / 'vocab.pt')
    else:
        vocab = Vocabulary()
        vocab.construct_vocabulary(sentences, labels)
        torch.save(vocab, model_path / 'vocab.pt')
    if (model_path / 'sentences.pt').is_file():
        train_dataset = torch.load(model_path / 'sentences.pt')
    else:
        train_dataset = PICDataset(sentences, labels, vocab)
        train_dataset.prepare_sentences()
        torch.save(train_dataset, model_path/ 'dataset.pt')

    a = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)

    net = build_model_34(language='EN', device=cfg['device'])

    net.train()

    print("Done")



if __name__ == '__main__':
    main()