import torch
from hw2.utils import read_dataset, evaluate_argument_identification, evaluate_argument_classification, print_table
import pathlib
from vocab import Vocabulary
from dataset import PICDataset
from torch.utils.data import DataLoader
from config import config as cfg
from implementation import build_model_34

curr_dir = pathlib.Path().absolute()
proj_dir = curr_dir.parent.parent
hw2_dir = curr_dir.parent
result_dir = curr_dir/'results'
model_dir = proj_dir/'model'
logs_dir = proj_dir/'logs'
data_train_path= proj_dir/'data'/'data_hw2'/'EN'/'train.json'
data_dev_path=proj_dir/'data'/'data_hw2'/'EN'/'dev.json'

def main():
    train_sentences, train_labels =read_dataset(data_train_path)
    val_sentences, val_labels = read_dataset(data_dev_path)
    if (model_dir / 'vocab.pt').is_file() and False:
        vocab = torch.load(model_dir / 'vocab.pt')
    else:
        vocab = Vocabulary()
        vocab.construct_vocabulary(train_sentences, train_labels)
        torch.save(vocab, model_dir / 'vocab.pt')
    if (model_dir / 'train_dataset.pt').is_file() and False:
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

    pathlib.Path('/tmp/sub1/sub2').mkdir(parents=True, exist_ok=True)

    result_dir.mkdir(exist_ok=True, parents=True)
    with open(result_dir/ ('results_' + cfg['model_name'][:-3] + '.txt') , 'w') as file:
        file.write(arg_id_results + '\n' + arg_clas_results)
    print("Done")



if __name__ == '__main__':
    main()