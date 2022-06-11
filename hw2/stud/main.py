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
    train_sentences, train_labels =read_dataset(data_train_path)
    val_sentences, val_labels = read_dataset(data_dev_path)
    if (model_path / 'vocab.pt').is_file():
        vocab = torch.load(model_path / 'vocab.pt')
    else:
        vocab = Vocabulary()
        vocab.construct_vocabulary(train_sentences, train_labels)
        torch.save(vocab, model_path / 'vocab.pt')
    if (model_path / 'train_sentences.pt').is_file():
        train_dataset = torch.load(model_path / 'train_dataset.pt')
        val_dataset = torch.load(model_path / 'val_dataset.pt')
    else:
        train_dataset = PICDataset(train_sentences, train_labels, vocab)
        train_dataset.prepare_sentences()
        torch.save(train_dataset, model_path/ 'train_dataset.pt')
        val_dataset = PICDataset(val_sentences, val_labels, vocab)
        val_dataset.prepare_sentences()
        torch.save(val_dataset, model_path/ 'val_dataset.pt')

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], collate_fn=train_dataset.collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['batch_size'], collate_fn=val_dataset.collate_fn, shuffle=True)

    net = build_model_34(language='EN', device=cfg['device'])
    net.predict([['']])
    net.SRLModel.train_net(train_dataloader, val_dataloader, model_path)

    print("Done")



if __name__ == '__main__':
    main()