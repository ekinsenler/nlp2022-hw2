import numpy as np
import torch
from hw2.utils import read_dataset, evaluate_argument_identification, evaluate_argument_classification, print_table
import pathlib
from stud.vocab import Vocabulary
from stud.dataset import PICDataset
from torch.utils.data import DataLoader
from stud.config import config as cfg
from stud.implementation import build_model_34
from transformers import AutoModel, AutoTokenizer
from stud.stud_util import create_conf_matrix, plot_class_dist
import matplotlib.pyplot as plt
curr_dir = pathlib.Path().absolute()
proj_dir = curr_dir.parent
stud_dir = curr_dir/'stud'
result_dir = stud_dir/'results'
model_dir = proj_dir/'model'
logs_dir = proj_dir/'logs'
data_train_path= proj_dir/'data'/'EN'/'train.json'
data_dev_path=proj_dir/'data'/'EN'/'dev.json'

def main():
    train_sentences, train_labels =read_dataset(data_train_path)
    val_sentences, val_labels = read_dataset(data_dev_path)
    if (model_dir / 'vocab.pt').is_file():
        vocab = torch.load(model_dir / 'vocab.pt')
        print(vocab.role_count)
        # plot_dict = vocab.role_count.copy()
        # plot_dict.pop('_')
        # plot_class_dist(plot_dict.values(), plot_dict.keys())
    else:
        vocab = Vocabulary()
        vocab.construct_vocabulary(train_sentences, train_labels)
        torch.save(vocab, model_dir / 'vocab.pt')
        print(vocab.role_count)
    if (model_dir / 'train_dataset.pt').is_file():
        train_dataset = torch.load(model_dir / 'train_dataset.pt')
        val_dataset = torch.load(model_dir / 'val_dataset.pt')
    else:
        checkpoint = "bert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, clean_text=False)
        model = AutoModel.from_pretrained(checkpoint)
        train_dataset = PICDataset(train_sentences, train_labels, vocab, bert_model=model, bert_tokenizer=tokenizer)
        train_dataset.prepare_sentences()
        torch.save(train_dataset, model_dir/ 'train_dataset.pt')
        val_dataset = PICDataset(val_sentences, val_labels, vocab, bert_model=model, bert_tokenizer=tokenizer)
        val_dataset.prepare_sentences()
        torch.save(val_dataset, model_dir/ 'val_dataset.pt')

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], collate_fn=train_dataset.collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['batch_size'], collate_fn=val_dataset.collate_fn, shuffle=True)

    net = build_model_34(language='EN', device='cuda')
    if cfg['is_train']:
        model_name = net.SRLModel.train_net(train_dataloader, val_dataloader, model_dir, logs_dir)
        cfg.set('model_name', model_name)
    else:
        net.SRLModel.load_state_dict(torch.load(model_dir / cfg['model_name'], map_location=torch.device('cpu')))

    prediction_val = net.predict_sentences(val_dataloader)
    arg_identification_val = evaluate_argument_identification(val_dataset.labels, prediction_val)
    arg_id_results = print_table('Argument Identification results on validation data', arg_identification_val)
    print(arg_id_results)
    arg_classification_val = evaluate_argument_classification(val_dataset.labels, prediction_val)
    arg_clas_results = print_table('Argument Classification results on validation data', arg_classification_val)
    print(arg_clas_results)

    #pathlib.Path('/tmp/sub1/sub2').mkdir(parents=True, exist_ok=True)

    result_dir.mkdir(exist_ok=True, parents=True)
    with open(result_dir/ ('results_' + cfg['model_name'][:-3] + '.txt') , 'w') as file:
        file.write(arg_id_results + '\n' + arg_clas_results)
    print("Done")

    conf_matrix, names = create_conf_matrix(val_labels, prediction_val)
    #sns.heatmap(conf_matrix, annot=True, annot_kws= names)



if __name__ == '__main__':
    main()