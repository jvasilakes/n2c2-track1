import sys
import bisect
import warnings
import itertools
from typing import List
from functools import partial

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import ConcatDataset, DataLoader

from src.data.base import BasicBertDataModule
from src.data.utils import register_sampler, SAMPLER_LOOKUP


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


class CombinedDataset(ConcatDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "CombinedDataset"

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
                 dataset_sample_strategy="proportional",
                 dataset_sampler_kwargs=None):
        super().__init__()
        # Sometimes this module can hit the system open file limit.
        # This seems to fix that.
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.datamodules = datamodules
        self.dataset_sample_strategy = dataset_sample_strategy
        self.dataset_sampler_kwargs = dataset_sampler_kwargs or {}
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
        dm_with_val = [dm for dm in self.datamodules if dm.val is not None]
        if len(dm_with_val) == 0:
            warnings.warn("No dev sets found.")
            self.val = None
        else:
            if len(dm_with_val) < len(self.datamodules):
                dm_wo_val_names = [dm.name for dm in self.datamodules
                                   if dm not in dm_with_val]
                warnings.warn(f"The following datasets do not have a dev set defined: {dm_wo_val_names}")  # noqa
            self.val = CombinedDataset([dm.val for dm in dm_with_val])

        dm_with_test = [dm for dm in self.datamodules if dm.test is not None]
        if len(dm_with_test) == 0:
            warnings.warn("No test sets found.")
            self.test = None
        else:
            if len(dm_with_test) < len(self.datamodules):
                dm_wo_test_names = [dm.name for dm in self.datamodules
                                    if dm not in dm_with_test]
                warnings.warn(f"The following datasets do not have a test set defined: {dm_wo_test_names}")  # noqa
            self.test = CombinedDataset([dm.test for dm in dm_with_test])

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
        sampler_class = SAMPLER_LOOKUP[self.dataset_sample_strategy]
        sampler = sampler_class(
            self.train, self.batch_size, shuffle_examples=True,
            **self.dataset_sampler_kwargs)
        train_collate_fn = partial(self.encode_and_collate, split="train")
        return DataLoader(self.train, collate_fn=train_collate_fn,
                          num_workers=4, batch_sampler=sampler)

    def val_dataloader(self, predicting=False):
        if self._ran_setup is False:
            raise ValueError("Run setup() first!")
        val_batch_size = self.batch_size
        if predicting is True:
            val_batch_size = 1
        sampler = SequentialDatasetSampler(
            self.val, val_batch_size, shuffle_examples=False, exhaust_all=True)
        # PyTorch Lightning does some weird stuff with dataset samplers
        # when running prediction, specifically re-instantiating the
        # sampler with some assumptions regarding its type. To avoid any
        # errors but satisfy the requirement that each batch must contain
        # examples from a single dataset, we have to use a batch size of
        # 1 and wrap IterativeDatasetSampler in a BatchSampler.
        if predicting is True:
            sampler = torch.utils.data.sampler.BatchSampler(
                    sampler, 1, drop_last=False)
        val_collate_fn = partial(self.encode_and_collate, split="val")
        return DataLoader(self.val, collate_fn=val_collate_fn,
                          num_workers=4, batch_sampler=sampler)

    def test_dataloader(self, predicting=False):
        if self._ran_setup is False:
            raise ValueError("Run setup() first!")
        if self.test is not None:
            test_batch_size = self.batch_size
            if predicting is True:
                test_batch_size = 1
            sampler = SequentialDatasetSampler(
                self.test, test_batch_size,
                shuffle_examples=False, exhaust_all=True)
            # PyTorch Lightning does some weird stuff with dataset samplers
            # when running prediction, specifically re-instantiating the
            # sampler with some assumptions regarding its type. To avoid any
            # errors but satisfy the requirement that each batch must contain
            # examples from a single dataset, we have to use a batch size of
            # 1 and wrap IterativeDatasetSampler in a BatchSampler.
            if predicting is True:
                sampler = torch.utils.data.sampler.BatchSampler(
                        sampler, 1, drop_last=False)
            test_collate_fn = partial(self.encoded_and_collate, split="test")
            return DataLoader(self.test, collate_fn=test_collate_fn,
                              num_workers=4, batch_sampler=sampler)
        return None

    def encode_and_collate(self, examples, split="train"):
        """
        Use the encode_and_collate function from the member datamodules.
        """
        datamodule_index = examples[0][0]
        # remove the dataset index
        examples = [example for (_, example) in examples]
        collate_fn = self.datamodules[datamodule_index].encode_and_collate
        collated = collate_fn(examples)
        split_datamods = [dm for dm in self.datamodules
                          if getattr(dm, split, None) is not None]
        collated["dataset"] = split_datamods[datamodule_index].name
        # change the name of the task key in labels to use the dataset name
        keys = list(collated["labels"].keys())
        for key in keys:
            new_key = f"{collated['dataset']}:{key}"
            collated["labels"][new_key] = collated["labels"][key]
            del collated["labels"][key]
        return collated


class DatasetSampler(torch.utils.data.sampler.Sampler):
    """
    The base strategy for sampling from a set of Datasets
    where every example in a batch must be from the same Dataset.

    Usage:
      sampler = DatasetSampler([dataset1, dataset2, ...], 16, **kwargs)
      sampler.setup()
      for batch_idxs in sampler:
         ...

    dataset: torch.utils.data.ConcatDataset
    batch_size: int
    shuffle_examples: whether to shuffle the examples in each dataset.
                      Default False.
    exhaust_all: whether to exhaust all datasets before returning a
                 StopIteration. If False, raises a StopIteration after
                 the first dataset is exhausted. Default True.
    """

    def __init__(
            self,
            dataset: ConcatDataset,
            batch_size,
            shuffle_examples=False,
            exhaust_all=True,
            name=None):
        self.name = name or "DatasetSampler"
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_examples = shuffle_examples
        self.exhaust_all = exhaust_all
        self.reset()

    def __iter__(self):
        # Reset valid datasets and batchers
        self.on_epoch_start()
        while True:
            if self.exhausted is True:
                self.on_exhaustion()
                break
            try:
                dataset_idx = self.sample_dataset_idx()
                yield self.sample_batch(dataset_idx)  # Raises StopIteration
                self.on_sample_batch_end()
            except StopIteration:
                self.rm_dataset(dataset_idx)

    def reset(self):
        self.valid_dataset_idxs = list(range(len(self.dataset.datasets)))
        self.batchers = self._get_batchers()

    def sample_dataset_idx(self):
        raise NotImplementedError()

    def sample_batch(self, dataset_idx):
        batch_idxs = next(self.batchers[dataset_idx])
        batch_idxs = [i for i in batch_idxs if i is not None]
        if dataset_idx > 0:
            offset = self.dataset.cumulative_sizes[dataset_idx - 1]
            batch_idxs = [i + offset for i in batch_idxs]
        return batch_idxs

    def rm_dataset(self, idx):
        """
        Remove the dataset at idx from this sampler.
        """
        try:
            self.valid_dataset_idxs.remove(idx)
        except ValueError:
            raise IndexError(f"No dataset at index {idx}.")

    @property
    def exhausted(self):
        if self.exhaust_all is True:
            if len(self.valid_dataset_idxs) == 0:
                return True
        else:
            if len(self.valid_dataset_idxs) < len(self.dataset.datasets):
                return True

    def on_epoch_start(self):
        """
        Called just before running __iter__.
        To be overridden in child classes.
        """
        pass

    def on_sample_batch_end(self):
        """
        Called just after self.sample_batch()
        To be overridden in child classes.
        """
        pass

    def on_exhaustion(self):
        """
        Called as soon as self.exhausted evaluates to True.
        To be overridden in child classes.
        """
        self.valid_dataset_idxs = list(range(len(self.dataset.datasets)))
        self.batchers = self._get_batchers()

    def _get_batchers(self):
        batchers = []
        for dataset in self.dataset.datasets:
            example_idxs = np.arange(len(dataset))
            if self.shuffle_examples is True:
                np.random.shuffle(example_idxs)
            batcher = self._grouper(example_idxs, self.batch_size)
            batchers.append(batcher)
        return batchers

    @staticmethod
    def _grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return itertools.zip_longest(fillvalue=fillvalue, *args)

    def __len__(self):
        lengths = [len(ds) for ds in self.dataset.datasets]
        return int(sum([np.ceil(ln / self.batch_size) for ln in lengths]))


@register_sampler("sequential")
class SequentialDatasetSampler(DatasetSampler):
    """
    Goes through the datasets in order. Note that the examples
    within each dataset will not be in order if shuffle_examples=True.
    """
    def __init__(self, *args, **kwargs):
        if "name" not in kwargs.keys():
            kwargs["name"] = "SequentialDatasetSampler"
        super().__init__(*args, **kwargs)

    def sample_dataset_idx(self):
        return self.valid_dataset_idxs[0]


@register_sampler("random")
class RandomDatasetSampler(DatasetSampler):

    def __init__(self, *args, **kwargs):
        if "name" not in kwargs.keys():
            kwargs["name"] = "RandomDatasetSampler"
        super().__init__(*args, **kwargs)

    def sample_dataset_idx(self):
        return np.random.choice(self.valid_dataset_idxs)


@register_sampler("weighted")
class WeightedDatasetSampler(DatasetSampler):
    """
    weights: None, 'lengths', or list of positive numbers representing the
             unnormalized weight for each dataset.
             If 'lengths', sets weights equal to the number of
             examples in each dataset.
    """
    def __init__(self, *args, weights=None, **kwargs):
        if "name" not in kwargs.keys():
            kwargs["name"] = "WeightedDatasetSampler"
        super().__init__(*args, **kwargs)
        self.weights = self._get_weights(weights)

    def sample_dataset_idx(self):
        probs = self.get_dataset_probs()
        return np.random.choice(self.valid_dataset_idxs, p=probs)

    def get_dataset_probs(self):
        valid_weights = self.weights[self.valid_dataset_idxs]
        probs = valid_weights / valid_weights.sum()
        return probs

    def _get_weights(self, weights):
        if weights is None:
            weights = np.ones(len(self.dataset.datasets))
            warnings.warn("No weights specified for WeightedDatasetSampler. Falling back to Uniform weights.")  # noqa
        elif weights == "lengths":
            weights = [len(ds) for ds in self.dataset.datasets]
            warnings.warn(f"Using dataset lengths as sampler weights: {weights}")  # noqa
        elif isinstance(weights, (list, np.ndarray)):
            if len(weights) != len(self.dataset.datasets):
                raise ValueError(f"Got different number of weights and datasets: {len(weights)}, {len(datasets)}")  # noqa
            weights = np.array(weights).astype(float)
            assert np.all(weights > 0.)
        else:
            raise ValueError(f"weights should be None or a list of positive numbers. Got {weights}.")  # noqa
        return np.array(weights)

    @property
    def exhausted(self):
        if len(self.valid_dataset_idxs) == 0:
            warnings.warn("Epoch ended early! This is expected occassionally, but not often!")  # noqa
            return True
        # if self.exhaust_all is True then self.target_samples_this_epoch
        #  is equal to the number of batches required to exhaust all datasets.
        if self.num_samples_this_epoch == self.target_samples_this_epoch:
            return True
        return False

    def on_epoch_start(self):
        super().on_epoch_start()
        self.target_samples_this_epoch = len(self)
        self.num_samples_this_epoch = 0

    def on_sample_batch_end(self):
        self.num_samples_this_epoch += 1

    def _estimate_length(self, lengths, probs):
        # Total number of steps
        nsteps = np.ceil(np.array(lengths) / self.batch_size).astype(int)
        n = nsteps.min()
        while n < (sys.maxsize / 2):  # A crude ceiling...
            dists = [stats.binom(n, p) for p in probs]
            # probabilities of getting between nsteps-var and nsteps successes
            # P[X > lb] and P[X < nsteps]
            variances = [int(np.ceil(d.var())) for d in dists]
            ps = np.array([1. - d.cdf(s - v) for (d, v, s)
                           in zip(dists, variances, nsteps)])
            # N.B. The threshold depends on the batch size, but this shouldn't
            #  matter too much in practice. E.g. a lower batch size requires
            #  a higher threshold.
            if any(ps >= 0.85):
                return n
            n += 1
        raise ValueError("WeightedDatasetSampler._estimate_length() exceeded sys.maxsize/2, which is very large! Check that your dataset probabilities make sense: {probs}")  # noqa

    def __len__(self):
        """
        Computing the length of a WeightedDatasetSampler is awkward because it
        samples from each dataset according to a probability and can end when
        it exhaustes just one. This means that its actual length might be
        unknown and changes every epoch.

        We address this by estimating the number of trials required to exhaust
        one of the datasets. This is done by computing P[D-c <= X <= D]
        under a set of binomial distributions of different n and determining
        the smallest n under which one of the datasets has P[D-c <= X <= D]
        >= 0.85. In other words, we want to know how many steps it takes to
        exhaust one of the datasets with probability 0.85 or greater. See
        _estimate_length for the implementation.
        """
        if getattr(self, "_len_cache", None) is None:
            self._len_cache = {}
        if self.exhaust_all is True:
            return super().__len__()
        # Compute weighted average of lengths
        lengths = np.array([len(ds) for ds in self.dataset.datasets])
        lengths = lengths[self.valid_dataset_idxs]
        probs = self.get_dataset_probs()
        cache_key = tuple([*lengths, *probs])
        try:
            nsteps = self._len_cache[cache_key]
        except KeyError:
            nsteps = self._estimate_length(lengths, probs)
            self._len_cache[cache_key] = nsteps
        return nsteps


@register_sampler("scheduled")
class ScheduledWeightedSampler(WeightedDatasetSampler):
    """
    Wraps a WeightedDatasetSampler to modify the
    weights according to the current epoch.

    max_steps: total number of steps. Equal to number of training epochs.
    num_cycles: number of times for the weights to cycle between
                minimum and maximum values in max_steps.
                E.g. num_cycles=1 means that the weights will end up where
                they began after max_steps.
    shift: A float specifying how much to shift the curves.
    invert: If True, starts from uniform weights.
            If False, starts from target weights.

    An approximation of the method implemented in
    "BERT and PALs: Projected Attention Layers for
     Efficient Adaptation in Multi-Task Learning"
    (Stickland and Murray, 2019)
    https://proceedings.mlr.press/v97/stickland19a.html
    is to set the weights of the sampler to the lengths of each dataset
    and set num_cycles=0.2.
    """
    def __init__(self, *sampler_args, max_steps=10,
                 num_cycles=0, shift=0.0, invert=False, **sampler_kwargs):
        if "name" not in sampler_kwargs.keys():
            sampler_kwargs["name"] = "ScheduledWeightedSampler"
        super().__init__(*sampler_args, **sampler_kwargs)
        if max_steps <= 1:
            raise ValueError("max_steps must be >= 1!")
        self.max_steps = max_steps
        self.num_cycles = num_cycles
        self.shift = shift
        self.invert = invert

    def reset(self):
        super().reset()
        self.step = 0

    def sample_dataset_idx(self, step=None):
        if step is None:
            step = self.step
        probs = self.get_dataset_probs(step=step)
        return np.random.choice(self.valid_dataset_idxs, p=probs)

    def get_dataset_probs(self, step=None, dataset_idxs=None):
        if step is None:
            step = self.step
        if dataset_idxs is None:
            dataset_idxs = self.valid_dataset_idxs
        valid_weights = self.weights[dataset_idxs]
        modded_weights = valid_weights**self.alpha(step=step)
        probs = modded_weights / modded_weights.sum()
        return probs

    def alpha(self, step=None):
        if step is None:
            step = self.step
        num = np.pi * step * self.num_cycles * 2
        trig_fn = np.cos
        if self.invert is True:
            trig_fn = np.sin
        shift = (self.max_steps - 1) * self.shift
        return trig_fn(num / (self.max_steps - 1) + shift)

    def on_exhaustion(self):
        super().on_exhaustion()
        self.step += 1

    def plot_schedule(self, steps=None, plot_lengths=False):
        """
        Use this to make sure ahead of time that you're using
        the schedule you want.
        """
        if steps is None:
            steps = self.max_steps
        all_idxs = list(range(len(self.dataset.datasets)))
        probs = np.array([self.get_dataset_probs(step=s, dataset_idxs=all_idxs)
                          for s in range(steps)])
        data_lengths = np.array([len(ds) for ds in self.dataset.datasets])
        nsteps = np.ceil(data_lengths / self.batch_size).astype(int)
        for c in range(probs.shape[1]):
            dataset_name = f"{self.dataset.datasets[c].name} (n={nsteps[c]})"
            style = ['-', '--', ':'][c % 3]
            plt.plot(probs[:, c], label=dataset_name, linestyle=style)
        if plot_lengths is True:
            if self.exhaust_all is True:
                lens = [super().__len__()] * steps
            else:
                lens = [self._estimate_length(data_lengths, self.get_dataset_probs(s))  # noqa
                        for s in range(steps)]
            norm_lens = [ln/max(lens) for ln in lens]
            plt.plot(norm_lens, color="red",
                     label=f"num_steps (max={max(lens)})")
        plt.ylim(-0.05, 1.05)
        plt.legend(loc="upper center", bbox_to_anchor=[0.5, 1.15],
                   ncol=2, shadow=True, fancybox=True)
        plt.show()


@register_sampler("stickland-murray")
class SticklandMurraySampler(ScheduledWeightedSampler):
    """
    The annealed sampling strategy proposed in
    "BERT and PALs: Projected Attention Layers for
     Efficient Adaptation in Multi-Task Learning"
    (Stickland and Murray, 2019)
    https://proceedings.mlr.press/v97/stickland19a.html
    """
    def __init__(self, *sampler_args, max_steps=10,
                 anneal_rate=0.8, **sampler_kwargs):
        if "name" not in sampler_kwargs.keys():
            sampler_kwargs["name"] = "SticklandMurraySampler"

        weights = "lengths"
        if "weights" in sampler_kwargs.keys():
            warnings.warn(f"SticklandAndMurraySampler: overriding provided weights {sampler_kwargs['weights']} with dataset lengths.")  # noqa
        sampler_kwargs["weights"] = weights
        super().__init__(*sampler_args, **sampler_kwargs)
        if max_steps <= 1:
            raise ValueError("max_steps must be >= 1!")
        self.max_steps = max_steps
        self.anneal_rate = anneal_rate

    def alpha(self, step=None):
        if step is None:
            step = self.step
        alpha = 1. - (self.anneal_rate * (step / (self.max_steps - 1)))
        return alpha


if __name__ == "__main__":
    print("Available dataset sampling methods")
    for (name, cls) in SAMPLER_LOOKUP.items():
        print(f"  {name}: {cls}")
