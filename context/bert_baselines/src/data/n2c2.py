import os
import warnings
from typing import List
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

import brat_reader as br

from src.data.base import BratMultiTaskDataset, BasicBertDataModule
from src.data.utils import register_dataset, DATASET_LOOKUP


class n2c2DataModule(BasicBertDataModule):

    @classmethod
    def from_config(cls, config, **override_kwargs):
        compute_class_weights = config.classifier_loss_kwargs.get("class_weights", None)  # noqa
        kwargs = {
            "dataset_name": config.dataset_name,
            "data_dir": config.data_dir,
            "sentences_dir": config.sentences_dir,
            "batch_size": config.batch_size,
            "bert_model_name_or_path": config.bert_model_name_or_path,
            "tasks_to_load": config.tasks_to_load,
            "max_seq_length": config.max_seq_length,
            "window_size": config.window_size,
            "max_train_examples": config.max_train_examples,
            "sample_strategy": config.sample_strategy,
            "compute_class_weights": compute_class_weights,
            "mark_entities": config.mark_entities,
            "entity_markers": config.entity_markers,
            "use_levitated_markers": config.use_levitated_markers,
            "levitated_pos_tags": config.levitated_pos_tags,
        }
        for (key, val) in override_kwargs.items():
            kwargs[key] = val
        return cls(**kwargs)

    def __init__(
            self, dataset_name, data_dir, sentences_dir,
            batch_size, bert_model_name_or_path,
            tasks_to_load="all",
            max_seq_length=128,
            window_size=0,
            sample_strategy=None,
            max_train_examples=-1,
            compute_class_weights=None,
            mark_entities=False,
            entity_markers=None,
            use_levitated_markers=False,
            levitated_pos_tags=None,
            name=None):

        if name is None:
            name = dataset_name
        super().__init__(
            bert_model_name_or_path,
            max_seq_length=max_seq_length,
            mark_entities=mark_entities,
            entity_markers=entity_markers,
            use_levitated_markers=use_levitated_markers,
            levitated_pos_tags=levitated_pos_tags,
            name=name)
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.sentences_dir = sentences_dir
        self.batch_size = batch_size
        self.bert_model_name_or_path = bert_model_name_or_path
        self.tasks_to_load = tasks_to_load
        self.window_size = window_size
        self.sample_strategy = sample_strategy
        self.max_train_examples = max_train_examples
        self.compute_class_weights = compute_class_weights
        self._ran_setup = False

    def setup(self, stage=None):
        load_labels = True
        if stage == "predict":
            load_labels = False
            warnings.warn("Predict mode ON: not loading gold-labels.")
        train_path = os.path.join(self.data_dir, "train")
        train_sent_path = os.path.join(self.sentences_dir, "train")
        self.train = self.dataset_class(
                train_path, train_sent_path,
                window_size=self.window_size,
                label_names=self.tasks_to_load,
                max_examples=self.max_train_examples,
                mark_entities=self.mark_entities,
                entity_markers=self.entity_markers,
                load_labels=load_labels)

        val_path = os.path.join(self.data_dir, "dev")
        val_sent_path = os.path.join(self.sentences_dir, "dev")
        if os.path.exists(val_path):
            self.val = self.dataset_class(
                    val_path, val_sent_path,
                    window_size=self.window_size,
                    label_names=self.tasks_to_load,
                    mark_entities=self.mark_entities,
                    entity_markers=self.entity_markers,
                    load_labels=load_labels)
        else:
            warnings.warn("No dev set found.")
            self.val = None

        test_path = os.path.join(self.data_dir, "test")
        test_sent_path = os.path.join(self.sentences_dir, "test")
        if os.path.exists(test_path):
            self.test = self.dataset_class(
                    test_path, test_sent_path,
                    window_size=self.window_size,
                    label_names=self.tasks_to_load,
                    mark_entities=self.mark_entities,
                    entity_markers=self.entity_markers,
                    load_labels=load_labels)
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
  name: {self.name},
  data_dir: {self.data_dir},
  sentences_dir: {self.sentences_dir},
  batch_size: {self.batch_size},
  bert_model_name_or_path: {self.bert_model_name_or_path},
  max_seq_length: {self.max_seq_length},
  tasks_to_load: {self.tasks_to_load},
  window_size: {self.window_size},
  max_train_examples: {self.max_train_examples},
  sample_strategy: {self.sample_strategy},
  class_weights: {self.class_weights},
  entity_markers: {self.entity_markers},
  use_levitated_markers: {self.use_levitated_markers}
  levitated_pos_tags: {self.levitated_pos_tags}"""

    @property
    def dataset_class(self):
        if "_dataset_class" in self.__dict__.keys():
            return self._dataset_class
        else:
            self._dataset_class = DATASET_LOOKUP[self.dataset_name]
            return self._dataset_class

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

    def inverse_transform(self, task=None, encoded_labels=None):
        """
        Just exposes inverse_transform from the underlying dataset.
        """
        return self.train.inverse_transform(
                task=task, encoded_labels=encoded_labels)

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
        if self.val is not None:
            return DataLoader(self.val, batch_size=self.batch_size,
                              collate_fn=self.encode_and_collate,
                              num_workers=4)
        return None

    def test_dataloader(self):
        if self.test is not None:
            return DataLoader(self.test, batch_size=self.batch_size,
                              collate_fn=self.encode_and_collate,
                              num_workers=4)
        return None

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


@register_dataset("n2c2Context", n2c2DataModule)
class n2c2ContextDataset(BratMultiTaskDataset):
    EXAMPLE_TYPE = "Disposition"
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


@register_dataset("n2c2Assertion", n2c2DataModule)
class n2c2AssertionDataset(BratMultiTaskDataset):
    EXAMPLE_TYPE = "Assertion"
    ENCODINGS = {
        # Task        label: encoding       num_examples
        "Assertion": {"absent": 0,        # 1594
                      "associated_with_someone_else": 1,  # 89
                      "conditional": 2,   # 73
                      "hypothetical": 3,  # 379
                      "possible": 4,      # 309
                      "present": 5}       # 4621
    }


@register_dataset("n2c2Assertion-Presence", n2c2DataModule)
class n2c2AssertionPresenceDataset(BratMultiTaskDataset):
    EXAMPLE_TYPE = "Assertion"
    ENCODINGS = {
        # Task        label: encoding       num_examples
        "Assertion": {"absent": 0,        # 1594
                      "present": 1}       # 4621
    }

    def filter_examples(self, examples: List[br.Annotation]):
        filtered = []
        for ex in examples:
            if ex.value in self.ENCODINGS["Assertion"].keys():
                filtered.append(ex)
        return filtered


@register_dataset("n2c2Assertion-Condition", n2c2DataModule)
class n2c2AssertionConditionDataset(BratMultiTaskDataset):
    EXAMPLE_TYPE = "Assertion"
    ENCODINGS = {
        # Task        label: encoding       num_examples
        "Assertion": {"conditional": 0,   # 73
                      "hypothetical": 1,  # 379
                      "possible": 2,      # 309
                      }
    }

    def filter_examples(self, examples: List[br.Annotation]):
        filtered = []
        for ex in examples:
            if ex.value in self.ENCODINGS["Assertion"].keys():
                filtered.append(ex)
        return filtered


@register_dataset("i2b2Event", n2c2DataModule)
class i2b2EventDataset(BratMultiTaskDataset):
    EXAMPLE_TYPE = "Disposition"
    ENCODINGS = {
            # Task        label: encoding      num_examples
            "Certainty": {"factual": 0,      # 100
                          "conditional": 1,  # 10 + 7 from "suggestion"
                          "unknown": 2,
                          },
            "Event": {"start": 0,            # 56 + 12 from "start-continue"
                      "stop": 1,             # 27
                      "continue": 2,         # 21 + 1 from "coninue"
                      "unknown": 3,
                      },
            "Temporality": {"past": 0,       # 88
                            "future": 1,     # 13
                            "present": 2,    # 11
                            "unknown": 3,
                            },
            }

    def preprocess_example(self, example: br.Annotation):
        """
        Certainty: suggestion -> conditional
        Event: start-continue -> start
        Event: coninue -> continue
        """
        # Some attributes are annotated "nm" for not mentioned, but
        # these are not included in the brat data, so we fill them
        # in as "unknown" here.
        for task in self.ENCODINGS.keys():
            if task not in example.attributes.keys():
                attr = br.Attribute(
                        _id=example.id.replace('E', 'A'),
                        _type=task,
                        value="unknown",
                        reference=example,
                        _source_file=example._source_file)
                example.attributes[task] = attr
        if example.attributes["Certainty"].value == "suggestion":
            example.attributes["Certainty"].value = "conditional"
        if example.attributes["Event"].value == "start-continue":
            example.attributes["Event"].value = "start"
        if example.attributes["Event"].value == "coninue":
            example.attributes["Event"].value = "continue"
        return example


if __name__ == "__main__":
    print("Available Datasets")
    for (name, cls) in DATASET_LOOKUP.items():
        print(f"  {name}: {cls}")
