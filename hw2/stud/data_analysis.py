from vocab import Vocabulary
from hw2.utils import read_dataset
import pathlib

curr_dir = pathlib.Path().absolute()
proj_dir = curr_dir.parent
data_train_path= proj_dir/'data'/'EN'/'train.json'
data_dev_path=proj_dir/'data'/'EN'/'dev.json'

train_sentences, train_labels = read_dataset(data_train_path)
vocab_train = Vocabulary()
vocab_train.construct_vocabulary(train_sentences, train_labels)

dev_sentences, dev_labels = read_dataset(data_dev_path)
vocab_dev = Vocabulary()
vocab_dev.construct_vocabulary(dev_sentences,dev_labels)

train_roles_set = set(vocab_train.id2role.values())
dev_roles_set = set(vocab_dev.id2role.values())

print('Number of roles that are in dev set but not present in train set: ',dev_roles_set - train_roles_set)
print('Done...')