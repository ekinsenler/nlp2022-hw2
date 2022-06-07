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
    train_data='',
    training_prediction='mixed_teacher_forcing',
    dynamic_tf=False,
    device='cpu',
    batch_size=16
)