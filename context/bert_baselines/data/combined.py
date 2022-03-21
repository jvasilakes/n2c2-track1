import bisect
import warnings
import itertools
from typing import List

import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader

from .base import BasicBertDataModule


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


class CombinedDataset(ConcatDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        """
        This is exactly the same as the __getitem__ code for ConcatDataset,
        but returns the dataset_idx so we know which one we're using.
        See https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#ConcatDataset  # noqa
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")  # noqa
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, self.datasets[dataset_idx][sample_idx]


class CombinedDataModule(BasicBertDataModule):
    """
    Combines multiple data modules into one.
    Allows datasets with different formats, tasks, etc.
    to be used by a single model in a multi-task learning
    setup.
    """

    def __init__(self, datamodules: List[BasicBertDataModule],
                 dataset_sample_strategy="proportional"):
        super().__init__()
        self.datamodules = datamodules
        self.dataset_sample_strategy = dataset_sample_strategy
        # These are populated in setup(), after checking
        # that they are all compatible.
        self.batch_size = None
        self.bert_model_name_or_path = None
        self.mark_entities = None
        self._ran_setup = False

    def setup(self, stage=None):
        for dm in self.datamodules:
            if dm._ran_setup is False:
                # Ignore warnings about missing test splits.
                # We'll handle that below.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dm.setup()
        self.train = CombinedDataset([dm.train for dm in self.datamodules])
        self.val = CombinedDataset([dm.val for dm in self.datamodules])
        test_sets = [dm.test for dm in self.datamodules if dm.test is not None]
        if len(test_sets) == 0:
            warnings.warn("No test sets found.")
            self.test = None
        else:
            if len(test_sets) < len(self.datamodules):
                warnings.warn("Some datasets do not have a test set defined.")
            self.test = CombinedDataset(test_sets)

        # Check that datamodule names don't conflict
        if len(self.datamodules) > 1:
            if all([dm.name == self.datamodules[0].name]):
                raise ValueError(f"Member BasicBertDataModules should have different names!")  # noqa

        # Check that all batch sizes are the same
        batch_sizes = [dm.batch_size for dm in self.datamodules]
        if all([bs == batch_sizes[0] for bs in batch_sizes]):
            self.batch_size = batch_sizes[0]
        else:
            raise ValueError(f"batch sizes differ! {batch_sizes}")
        # Check that all bert models are the same
        berts = [dm.bert_model_name_or_path for dm in self.datamodules]
        if all([bert == berts[0] for bert in berts]):
            self.bert_model_name_or_path = berts[0]
        else:
            raise ValueError(f"bert models differ! {berts}")
        # Check that entity marking is the same
        marks = [dm.mark_entities for dm in self.datamodules]
        if all([mark == marks[0] for mark in marks]):
            self.mark_entities = marks[0]
        else:
            raise ValueError(f"mark_entites differ! {marks}")
        self._ran_setup = True

    def __str__(self):
        return f"""{self.__class__}
  datamodules: {[dm.name for dm in self.datamodules]},
  batch_size: {self.batch_size},
  bert_model_name_or_path: {self.bert_model_name_or_path},
  dataset_sample_strategy: {self.dataset_sample_strategy}"""

    @property
    def label_spec(self):
        if self._ran_setup is False:
            raise ValueError("Run setup() first!")
        if getattr(self, "_label_spec", None) is not None:
            return self._label_spec

        spec = {}
        for dm in self.datamodules:
            for (task, label_size) in dm.label_spec.items():
                dm_task_str = f"{dm.name}:{task}"
                spec[dm_task_str] = label_size
        self._label_spec = spec
        return self._label_spec

    def inverse_transform(self, task, encoded):
        dataset = self.get_dataset_from_task(task)
        task_wo_dataset_name = task.split(':')[1]
        return dataset.inverse_transform(task_wo_dataset_name, encoded)

    def get_dataset_from_task(self, target_task, split="train"):
        if split not in ["train", "dev", "test"]:
            raise ValueError(f"Unknown split '{split}'")
        target_name = target_task.split(':')[0]
        for dm in self.datamodules:
            if dm.name == target_name:
                return getattr(dm, split)
        raise KeyError(target_task)

    def train_dataloader(self):
        if self._ran_setup is False:
            raise ValueError("Run setup() first!")
        sampler = ProportionalSampler(
            self.train, strategy=self.dataset_sample_strategy,
            batch_size=self.batch_size)
        return DataLoader(self.train, collate_fn=self.encode_and_collate,
                          num_workers=4, batch_sampler=sampler)

    def val_dataloader(self, predicting=False):
        if self._ran_setup is False:
            raise ValueError("Run setup() first!")
        sampler = IterativeDatasetSampler(
                self.val, self.batch_size, predicting=predicting)
        if predicting is True:
            sampler = torch.utils.data.sampler.BatchSampler(
                    sampler, 1, drop_last=False)
        return DataLoader(self.val, collate_fn=self.encode_and_collate,
                          num_workers=4, batch_sampler=sampler)

    def test_dataloader(self):
        if self._ran_setup is False:
            raise ValueError("Run setup() first!")
        if self.test is not None:
            sampler = IterativeDatasetSampler(self.test, self.batch_size)
            return DataLoader(self.test, collate_fn=self.encode_and_collate,
                              num_workers=4, batch_sampler=sampler)
        return None

    def encode_and_collate(self, examples):
        """
        Use the encode_and_collate function from the member datamodules.
        """
        datamodule_index = examples[0][0]
        # remove the dataset index
        examples = [example for (_, example) in examples]
        collate_fn = self.datamodules[datamodule_index].encode_and_collate
        collated = collate_fn(examples)
        collated["dataset"] = self.datamodules[datamodule_index].name
        # change the name of the task key in labels to use the dataset name
        keys = list(collated["labels"].keys())
        for key in keys:
            new_key = f"{collated['dataset']}:{key}"
            collated["labels"][new_key] = collated["labels"][key]
            del collated["labels"][key]
        return collated


class ProportionalSampler(torch.utils.data.sampler.Sampler):
    """
    Samples from each Dataset of a ConcatDataset proportional
    to their sizes. Each batch must come from the same dataset.

    If strategy=='annealed', uses the strategy proposed in
    "BERT and PALs: Projected Attention Layers for
     Efficient Adaptation in Multi-Task Learning"
    (Stickland and Murray, 2019)
    https://proceedings.mlr.press/v97/stickland19a.html
    """

    def __init__(self, dataset: ConcatDataset,
                 strategy="proportional", batch_size=16, **kwargs):
        self.dataset = dataset
        self.strategy = strategy
        self.batch_size = batch_size

        self.num_datasets = len(dataset.datasets)
        self.max_epochs = 10
        self.epoch = 0

    def __iter__(self):
        """
        Because we're sampling from each dataset according to a
        probability, we'll probably exhaust one before the rest.
        To address this, we keep track of all non-exhausted datasets
        and only sample from them each time.
        """
        not_exhausted = np.arange(self.num_datasets)
        dataset_lengths = [len(ds) for ds in self.dataset.datasets]
        idxs = [torch.randperm(length) for length in dataset_lengths]
        groupers = [grouper(shuffled_idxs, self.batch_size)
                    for shuffled_idxs in idxs]
        while True:
            try:
                probs = self.get_dataset_probs(
                    [self.dataset.datasets[i] for i in not_exhausted])
                dataset_idx = np.random.choice(
                    not_exhausted, p=probs)
                batch_idxs = next(groupers[dataset_idx])
                batch_idxs = [i for i in batch_idxs if i is not None]
                if dataset_idx > 0:
                    offset = self.dataset.cumulative_sizes[dataset_idx - 1]
                    batch_idxs = [i + offset for i in batch_idxs]
                yield batch_idxs
            except StopIteration:
                not_exhausted = [i for i in not_exhausted if i != dataset_idx]
                if len(not_exhausted) == 0:
                    self.epoch += 1
                    break

    def get_dataset_probs(self, datasets=None):
        if datasets is None:
            datasets = self.dataset.datasets
        lengths = [len(ds) for ds in datasets]
        if self.strategy == "proportional":
            probs = [ln/sum(lengths) for ln in lengths]
        elif self.strategy == "annealed":
            alpha = 1 - (0.8 * ((self.epoch - 1) / (self.max_epochs - 1)))
            lengths = [ln**alpha for ln in lengths]
            probs = [ln/sum(lengths) for ln in lengths]
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")
        return probs

    def __len__(self):
        lengths = [len(ds) for ds in self.dataset.datasets]
        return int(sum([np.ceil(ln / self.batch_size) for ln in lengths]))


class IterativeDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Each batch must come from the same dataset.
    """
    def __init__(self, dataset: ConcatDataset,
                 batch_size=16, predicting=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.predicting = predicting
        self.num_datasets = len(dataset.datasets)

    def __iter__(self):
        for (dataset_idx, dataset) in enumerate(self.dataset.datasets):
            idxs = np.arange(len(dataset))
            if dataset_idx > 0:
                idx_offset = self.dataset.cumulative_sizes[dataset_idx - 1]
            else:
                idx_offset = 0
            for batch_idxs in grouper(idxs, self.batch_size):
                batch_idxs = [i + idx_offset for i in batch_idxs
                              if i is not None]
                if self.predicting is True:
                    yield from batch_idxs
                else:
                    yield batch_idxs

    def __len__(self):
        lengths = [len(ds) for ds in self.dataset.datasets]
        return int(sum([np.ceil(ln / self.batch_size) for ln in lengths]))

    def str(self):
        return "IterativeDatasetSampler"
