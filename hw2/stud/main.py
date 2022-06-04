from hw2.utils import read_dataset
import pathlib
from vocab import Vocabulary

curr_dir = pathlib.Path(__file__)
proj_dir = curr_dir.parent.parent.parent
hw1_dir = curr_dir.parent.parent

data_train_path= proj_dir/'data'/'data_hw2'/'EN'/'train.json'
data_dev_path=proj_dir/'data'/'data_hw2'/'EN'/'dev.json'
model_path = proj_dir/'model'/'state_dict_model.pt'

def main():
    data=read_dataset(data_train_path)
    vocab = Vocabulary()
    vocab.build_vocab(data)

if __name__ == '__main__':
    main()