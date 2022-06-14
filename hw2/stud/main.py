import torch
from hw2.utils import read_dataset, evaluate_argument_identification, evaluate_argument_classification, print_table
import pathlib
from vocab import Vocabulary
from dataset import PICDataset
from torch.utils.data import DataLoader
from config import config as cfg
from implementation import build_model_34

curr_dir = pathlib.Path(__file__)
proj_dir = curr_dir.parent.parent.parent
hw1_dir = curr_dir.parent.parent
model_dir = proj_dir/'model'
data_train_path= proj_dir/'data'/'data_hw2'/'EN'/'train.json'
data_dev_path=proj_dir/'data'/'data_hw2'/'EN'/'dev.json'

def main():
    train_sentences, train_labels =read_dataset(data_train_path)
    val_sentences, val_labels = read_dataset(data_dev_path)
    if (model_dir / 'vocab.pt').is_file():
        vocab = torch.load(model_dir / 'vocab.pt')
    else:
        vocab = Vocabulary()
        vocab.construct_vocabulary(train_sentences, train_labels)
        torch.save(vocab, model_dir / 'vocab.pt')
    if (model_dir / 'train_sentences.pt').is_file():
        train_dataset = torch.load(model_dir / 'train_dataset.pt')
        val_dataset = torch.load(model_dir / 'val_dataset.pt')
    else:
        train_dataset = PICDataset(train_sentences, train_labels, vocab)
        train_dataset.prepare_sentences()
        torch.save(train_dataset, model_dir/ 'train_dataset.pt')
        val_dataset = PICDataset(val_sentences, val_labels, vocab)
        val_dataset.prepare_sentences()
        torch.save(val_dataset, model_dir/ 'val_dataset.pt')

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], collate_fn=train_dataset.collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['batch_size'], collate_fn=val_dataset.collate_fn, shuffle=True)

    net = build_model_34(language='EN', device='cuda')
    if cfg['is_train']:
        net.SRLModel.train_net(train_dataloader, val_dataloader, model_dir)
    else:
        net.SRLModel.load_state_dict(torch.load(model_dir / 'model_2epoch_1655215727.326815.pt', map_location=torch.device('cpu')))

    prediction_val = net.predict_sentences(val_dataloader)
    arg_identification_val = evaluate_argument_identification(val_dataset.labels, prediction_val)
    print_table('argIdentVal', arg_identification_val)
    arg_classification_val = evaluate_argument_classification(val_dataset.labels, prediction_val)
    print_table('argClassVal', arg_classification_val)

    print("Done")



if __name__ == '__main__':
    main()