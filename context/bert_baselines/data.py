import os
import json
import bisect
import warnings
import itertools
from glob import glob
from collections import Counter, defaultdict
from typing import List, Dict, Union

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import (Dataset, DataLoader,
                              WeightedRandomSampler, ConcatDataset)
from transformers import AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight

from brat_reader import BratAnnotations


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
            cls_name = cls_str.split('.')[1]
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


class n2c2ContextDataset(Dataset):

    ENCODINGS = {
            'Action': {'Decrease': 0,
                       'Increase': 1,
                       'OtherChange': 2,
                       'Start': 3,
                       'Stop': 4,
                       'UniqueDose': 5,
                       'Unknown': 6},
            'Actor': {'Patient': 0, 'Physician': 1, 'Unknown': 2},
            'Certainty': {'Certain': 0,
                          'Conditional': 1,
                          'Hypothetical': 2,
                          'Unknown': 3},
            'Negation': {'Negated': 0, 'NotNegated': 1},
            'Temporality': {'Future': 0,
                            'Past': 1,
                            'Present': 2,
                            'Unknown': 3}
            }
    SORTED_ATTRIBUTES = ["Action", "Actor", "Certainty",
                         "Negation", "Temporality"]

    def __init__(self, data_dir, sentences_dir,
                 label_names="all", window_size=0,
                 max_examples=-1, mark_entities=False):
        self.data_dir = data_dir
        self.sentences_dir = sentences_dir
        self.label_names = label_names
        self.window_size = window_size
        self.max_examples = max_examples
        self.mark_entities = mark_entities

        self.events, self.docids_to_texts = self._get_dispositions_and_texts()
        if label_names == "all":
            self.label_names = self.SORTED_ATTRIBUTES
        else:
            for name in self.label_names:
                if name not in self.SORTED_ATTRIBUTES:
                    raise ValueError(f"Unknown attribute {name}")
            self.label_names = sorted(label_names)

        self._inverse_encodings = self._invert_encodings()

    def inverse_transform(self, task, encoded_labels):
        return [self._inverse_encodings[task][enc_lab]
                for enc_lab in encoded_labels]

    def __len__(self):
        if self.max_examples == -1:
            return len(self.events)
        else:
            return len(self.events[:self.max_examples])

    def __getitem__(self, idx):
        # Get the sentences in the window
        event = self.events[idx]
        sentences = self.docids_to_texts[event.docid]
        si = event.sent_index
        start_sent = max([0, si - self.window_size])
        end_sent = min([len(sentences), si + self.window_size + 1])
        context = sentences[start_sent:end_sent]

        # Construct contiguous text, keeping character offsets consistent.
        text = context[0]["_text"]
        prev_sent = context[0]
        for sent in context[1:]:
            text += ' ' * (sent["start_index"] - prev_sent["end_index"])
            text += sent["_text"]
            prev_sent = sent

        # Compute the relative offset of the entity in the context
        entity_start = event.span.start_index - context[0]["start_index"]
        entity_end = event.span.end_index - context[0]["start_index"]
        # Surround the entity spans with '@'. E.g., 'He took @Toradol@'.
        if self.mark_entities is True:
            marked_text = text[:entity_start]
            marked_text += '@' + text[entity_start:entity_end] + '@'
            marked_text += text[entity_end:]
            text = marked_text
            entity_end += 2

        # Encode the labels
        labels = {}
        for attr_name in self.SORTED_ATTRIBUTES:
            if attr_name in self.label_names:
                attr = event.attributes[attr_name]
                val = attr.value
                labels[attr_name] = self.ENCODINGS[attr_name][val]

        return {"text": text,
                "entity_span": (entity_start, entity_end),
                # For reconstructing original entity span offsets
                #  in the source document.
                "char_offset": context[0]["start_index"],
                "labels": labels,
                "docid": event.docid}

    def _get_dispositions_and_texts(self):
        """
        Pair each disposition event with a sentence
        in a document. Adds the "docid" and "sent_index"
        fields to the disposition.
        """
        all_dispositions = []
        docids_to_texts = {}
        ann_glob = os.path.join(self.data_dir, "*.ann")
        ann_files = glob(ann_glob)
        if ann_files == []:
            raise OSError(f"No annotations found at {ann_glob}")
        for ann_file in glob(ann_glob):
            anns = BratAnnotations.from_file(ann_file)
            dispositions = anns.get_events_by_type("Disposition")
            docid = os.path.basename(ann_file).strip(".ann")
            for d in dispositions:
                d.update("docid", docid)
            all_dispositions.extend(dispositions)

            txt_file = os.path.join(self.sentences_dir, f"{docid}.txt.json")
            with open(txt_file, 'r') as inF:
                sent_data = [json.loads(line) for line in inF]
            # list of indices in sent_data, one per disposition.
            sent_idxs = self._match_events_to_sentences(
                    dispositions, sent_data)
            for (d, si) in zip(dispositions, sent_idxs):
                d.update("sent_index", si)
            docids_to_texts[docid] = sent_data
        return all_dispositions, docids_to_texts

    def _match_events_to_sentences(self, events, sentences):
        # Each character index in the full document maps to an
        #  index in sentences.
        sent_index_lookup = {}
        for (i, sent) in enumerate(sentences):
            for j in range(sent["start_index"], sent["end_index"]):
                sent_index_lookup[j] = i

        sent_idxs = []
        for e in events:
            try:
                sent_i = sent_index_lookup[e.span.start_index]
            except KeyError:
                print(f"MISSING {e}")
                continue
            sent_idxs.append(sent_i)
        return sent_idxs

    def _invert_encodings(self):
        inv_enc = {}
        for (task, label_encs) in self.ENCODINGS.items():
            inv_enc[task] = {}
            for (label, enc) in label_encs.items():
                inv_enc[task][enc] = label
        return inv_enc


class n2c2ContextDataModule(BasicBertDataModule):

    @classmethod
    def from_config(cls, config):
        compute_class_weights = config.classifier_loss_kwargs.get("class_weights", None)  # noqa
        return cls(
                config.data_dir,
                config.sentences_dir,
                batch_size=config.batch_size,
                bert_model_name_or_path=config.bert_model_name_or_path,
                tasks_to_load=config.tasks_to_load,
                max_seq_length=config.max_seq_length,
                window_size=config.window_size,
                max_train_examples=config.max_train_examples,
                sample_strategy=config.sample_strategy,
                compute_class_weights=compute_class_weights,
                mark_entities=config.mark_entities,
                )

    def __init__(self, data_dir, sentences_dir, batch_size,
                 bert_model_name_or_path, tasks_to_load="all",
                 max_seq_length=128, window_size=0, sample_strategy=None,
                 max_train_examples=-1, compute_class_weights=None,
                 mark_entities=False, name=None):
        super().__init__(name=name)
        self.data_dir = data_dir
        self.sentences_dir = sentences_dir
        self.batch_size = batch_size
        self.bert_model_name_or_path = bert_model_name_or_path
        self.tasks_to_load = tasks_to_load
        self.max_seq_length = max_seq_length
        self.window_size = window_size
        self.sample_strategy = sample_strategy
        self.max_train_examples = max_train_examples
        self.compute_class_weights = compute_class_weights
        self.mark_entities = mark_entities

        self.tokenizer = AutoTokenizer.from_pretrained(
                self.bert_model_name_or_path, use_fast=True)
        self._ran_setup = False

    def setup(self, stage=None):
        train_path = os.path.join(self.data_dir, "train")
        train_sent_path = os.path.join(self.sentences_dir, "train")
        self.train = n2c2ContextDataset(
                train_path, train_sent_path,
                window_size=self.window_size,
                label_names=self.tasks_to_load,
                max_examples=self.max_train_examples,
                mark_entities=self.mark_entities)

        val_path = os.path.join(self.data_dir, "dev")
        val_sent_path = os.path.join(self.sentences_dir, "dev")
        self.val = n2c2ContextDataset(
                val_path, val_sent_path,
                window_size=self.window_size,
                label_names=self.tasks_to_load,
                mark_entities=self.mark_entities)

        test_path = os.path.join(self.data_dir, "test")
        test_sent_path = os.path.join(self.sentences_dir, "test")
        if os.path.exists(test_path):
            self.test = n2c2ContextDataset(
                    test_path, test_sent_path,
                    window_size=self.window_size,
                    label_names=self.tasks_to_load,
                    mark_entities=self.mark_entities)
        else:
            warnings.warn("No test set found.")
            self.test = None

        if self.sample_strategy is None:
            self.sampler = None
        elif self.sample_strategy == "weighted":
            # This should give a near-uniform distribution of task
            # values across examples.
            weights = self._compute_sample_weights(self.train)
            self.sampler = WeightedRandomSampler(weights, len(weights))
        else:
            msg = f"Unknown sampling strategy {self.sample_strategy}"
            raise ValueError(msg)

        self._ran_setup = True

    def __str__(self):
        return f"""{self.__class__}
  data_dir: {self.data_dir},
  sentences_dir: {self.sentences_dir},
  batch_size: {self.batch_size},
  bert_model_name_or_path: {self.bert_model_name_or_path},
  tasks_to_load: {self.tasks_to_load},
  max_seq_length: {self.max_seq_length},
  window_size: {self.window_size},
  max_train_examples: {self.max_train_examples},
  sample_strategy: {self.sample_strategy},
  class_weights: {self.class_weights},
  mark_entities: {self.mark_entities}"""

    @property
    def label_spec(self):
        if self._ran_setup is False:
            raise ValueError("Run setup() first!")
        if getattr(self, "_label_spec", None) is not None:
            return self._label_spec

        spec = {task: len(encs) for (task, encs)
                in self.train.ENCODINGS.items()}
        if self.tasks_to_load != "all":
            spec = {task: n for (task, n) in spec.items()
                    if task in self.tasks_to_load}
        self._label_spec = spec
        return self._label_spec

    @property
    def class_weights(self):
        if self._ran_setup is False:
            raise ValueError("Run setup() first!")
        if getattr(self, "_class_weights", None) is not None:
            return self._class_weights

        if self.compute_class_weights is None:
            self._class_weights = None
        elif self.compute_class_weights == "balanced":
            self._class_weights = self._compute_class_weights(self.train)
        else:
            raise ValueError(f"Unsupported class weighting {self.compute_class_weights}")  # noqa
        return self._class_weights

    def train_dataloader(self):
        shuffle = True if self.sampler is None else False
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=shuffle, collate_fn=self.encode_and_collate,
                          num_workers=4, sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size,
                          collate_fn=self.encode_and_collate, num_workers=4)

    def test_dataloader(self):
        if self.test is not None:
            return DataLoader(self.test, batch_size=self.batch_size,
                              collate_fn=self.encode_and_collate,
                              num_workers=4)
        return None

    def encode_and_collate(self, examples):
        batch = {"encodings": None,
                 "entity_spans": [],
                 "char_offsets": [],
                 "texts": [],
                 "labels": defaultdict(list),
                 "docids": []
                 }

        for ex in examples:
            batch["entity_spans"].append(ex["entity_span"])
            batch["char_offsets"].append(ex["char_offset"])
            batch["texts"].append(ex["text"])
            batch["docids"].append(ex["docid"])
            for (task, val) in ex["labels"].items():
                batch["labels"][task].append(val)

        encodings = self.tokenizer(batch["texts"], truncation=True,
                                   max_length=self.max_seq_length,
                                   padding="max_length",
                                   return_offsets_mapping=True,
                                   return_tensors="pt")
        batch["encodings"] = encodings
        batch["entity_spans"] = torch.tensor(batch["entity_spans"])
        for task in batch["labels"].keys():
            batch["labels"][task] = torch.tensor(batch["labels"][task])
        return batch

    # Used with WeightedRandomSampler in setup()
    def _compute_sample_weights(self, train_dataset):
        task_vals = [tuple(ex["labels"][task]
                           for task in train_dataset.label_names)
                     for ex in train_dataset]
        val_counts = Counter(task_vals)
        val_weights = {val: 1. / val_counts[val] for val in val_counts.keys()}
        sample_weights = [val_weights[task_val] for task_val in task_vals]
        return sample_weights

    # Weights for each task can be passed to CrossEntropyLoss in a model.
    def _compute_class_weights(self, train_dataset):
        y_per_task = defaultdict(list)
        for ex in train_dataset:
            for (task, val) in ex["labels"].items():
                y_per_task[task].append(val)
        class_weights_per_task = {}
        for (task, task_y) in y_per_task.items():
            classes = list(sorted(set(task_y)))
            weights = compute_class_weight(
                    "balanced", classes=classes, y=task_y)
            class_weights_per_task[task] = torch.tensor(
                    weights, dtype=torch.float)
        return class_weights_per_task


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
                 sample_strategy="proportional"):
        super().__init__()
        self.datamodules = datamodules
        self.sample_strategy = sample_strategy
        # These are populated in setup(), after checking
        # that they are all compatible.
        self.batch_size = None
        self.bert_model_name_or_path = None
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
        self._ran_setup = True

    def __str__(self):
        return f"""{self.__class__}
  datasets: {[dm.name for dm in self.datamodules]},
  batch_size: {self.batch_size},
  bert_model_name_or_path: {self.bert_model_name_or_path},
  sample_strategy: {self.sample_strategy}"""

    @property
    def label_spec(self):
        if self._ran_setup is False:
            raise ValueError("Run setup() first!")
        if getattr(self, "_label_spec", None) is not None:
            return self._label_spec

        spec = {}
        for dm in self.datamodules:
            for (task, label_size) in dm.label_spec.items():
                dm_task_str = f"{dm.name}-{task}"
                spec[dm_task_str] = label_size
        self._label_spec = spec
        return self._label_spec

    def train_dataloader(self):
        if self._ran_setup is False:
            raise ValueError("Run setup() first!")
        sampler = ProportionalSampler(
            self.train, strategy=self.sample_strategy,
            batch_size=self.batch_size)
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=False, collate_fn=self.encode_and_collate,
                          num_workers=4, sampler=sampler)

    def val_dataloader(self):
        if self._ran_setup is False:
            raise ValueError("Run setup() first!")
        return DataLoader(self.val, batch_size=self.batch_size,
                          collate_fn=self.encode_and_collate, num_workers=4)

    def test_dataloader(self):
        if self._ran_setup is False:
            raise ValueError("Run setup() first!")
        if self.test is not None:
            return DataLoader(self.test, batch_size=self.batch_size,
                              collate_fn=self.encode_and_collate,
                              num_workers=4)
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
            new_key = f"{collated['dataset']}-{key}"
            collated["labels"][new_key] = collated["labels"][key]
            del collated["labels"][key]
        return collated


class ProportionalSampler(torch.utils.data.sampler.Sampler):
    """
    Samples from each Dataset of a ConcatDataset proportional
    to their sizes. If strategy=='annealed', uses the strategy
    proposed in
    "BERT and PALs: Projected Attention Layers for
     Efficient Adaptation in Multi-Task Learning"
    (Stickland and Murray, 2019)
    https://proceedings.mlr.press/v97/stickland19a.html
    """

    def __init__(self, dataset: ConcatDataset,
                 strategy="proportional", batch_size=16):
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
        groupers = [self.grouper(shuffled_idxs, self.batch_size)
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
                yield from batch_idxs
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

    @staticmethod
    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return itertools.zip_longest(fillvalue=fillvalue, *args)
