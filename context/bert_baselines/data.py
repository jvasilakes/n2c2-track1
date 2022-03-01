import os
import json
import warnings
from glob import glob
from collections import Counter, defaultdict

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight

from brat_reader import BratAnnotations


class n2c2SentencesDataset(Dataset):

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


class n2c2SentencesDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, sentences_dir, batch_size,
                 bert_model_name_or_path, tasks_to_load="all",
                 max_seq_length=128, window_size=0, sample_strategy="none",
                 max_train_examples=-1, compute_class_weights="none",
                 mark_entities=False):
        super().__init__()
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
        self.train = n2c2SentencesDataset(
                train_path, train_sent_path,
                window_size=self.window_size,
                label_names=self.tasks_to_load,
                max_examples=self.max_train_examples,
                mark_entities=self.mark_entities)

        val_path = os.path.join(self.data_dir, "dev")
        val_sent_path = os.path.join(self.sentences_dir, "dev")
        self.val = n2c2SentencesDataset(
                val_path, val_sent_path,
                window_size=self.window_size,
                label_names=self.tasks_to_load,
                mark_entities=self.mark_entities)

        test_path = os.path.join(self.data_dir, "test")
        test_sent_path = os.path.join(self.sentences_dir, "test")
        if os.path.exists(test_path):
            self.test = n2c2SentencesDataset(
                    test_path, test_sent_path,
                    window_size=self.window_size,
                    label_names=self.tasks_to_load,
                    mark_entities=self.mark_entities)
        else:
            warnings.warn("No test set found.")
            self.test = None

        if self.sample_strategy == "none":
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

        if self.compute_class_weights == "none":
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
