from hw2.utils import read_dataset
import pathlib
from vocab import Vocabulary
from dataset import PICDataset

curr_dir = pathlib.Path(__file__)
proj_dir = curr_dir.parent.parent.parent
hw1_dir = curr_dir.parent.parent

data_train_path= proj_dir/'data'/'data_hw2'/'EN'/'train.json'
data_dev_path=proj_dir/'data'/'data_hw2'/'EN'/'dev.json'
model_path = proj_dir/'model'/'state_dict_model.pt'

def main():
    sentences, labels =read_dataset(data_train_path)
    #vocab = Vocabulary()
    #vocab.build_vocab(sentences, labels)
    dataset = PICDataset(sentences, labels)
    dataset.prepare_data_points()

    print("Done")

if __name__ == '__main__':
    main()