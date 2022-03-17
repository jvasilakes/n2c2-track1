from typing import Dict, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BasicBertDataModule(pl.LightningDataModule):
    """
    The base data module for all BERT-based sequence classification datasets.
    """
    def __init__(self, name=None):
        self._name = name

    @property
    def name(self) -> str:
        if self._name is None:
            cls_str = str(self.__class__)
            cls_name = cls_str.split('.')[-1]
            self._name = cls_name.replace("DataModule'>", '')
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def label_spec(self) -> Dict[str, int]:
        """
        A mapping from task names to number of unique labels.
        E.g.,
        {"Negation": 2,
         "Certainty": 4}
        """
        raise NotImplementedError()

    def setup(self):
        """
        See the PyTorch Lightning docs at
        https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
        """
        raise NotImplementedError()

    def train_dataloader(self) -> DataLoader:
        """
        See the PyTorch Lightning docs at
        https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
        """
        raise NotImplementedError()

    def val_dataloader(self) -> DataLoader:
        """
        See the PyTorch Lightning docs at
        https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
        """
        raise NotImplementedError()

    def test_dataloader(self) -> Union[DataLoader, type(None)]:
        """
        See the PyTorch Lightning docs at
        https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
        """
        raise NotImplementedError()

    def encode_and_collate(self, examples) -> Dict:
        """
        Your own, unique function for combining a list of examples
        into a batch.
        """
        raise NotImplementedError()
