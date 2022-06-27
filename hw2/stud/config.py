import torch

class Config(dict):
    def __init__(self, **kwargs):
        """
        Initialize an instance of this class.
        Args:
        """
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set(self, key, value):
        """
        Sets the value to the value.
        Args:
            key: (str):
            value:
        """
        self[key] = value
        setattr(self, key, value)


config = Config(
    model_name='model_20epoch_1655302145.592678.pt',
    train_data='',
    training_prediction='mixed_teacher_forcing',
    dynamic_tf=False,
    batch_size=64,
    word_embed_dim=200,
    pretrained_embed='glove-wiki-gigaword-200',
    lemma_embed_dim=200,
    pos_embed_dim=200,
    pred_embed_dim=200,
    bert_embed_dim=768,
    bert_bilstm_dim=200,
    hidden_size=500,
    num_epoch=20,
    lr=1e-4,
    log_dir='',
    model_path='',
    dropout=0.2,
    num_layers=2,
    is_train = True
)