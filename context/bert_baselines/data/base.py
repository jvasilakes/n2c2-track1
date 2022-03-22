import os
import json
from glob import glob
from typing import Dict, Union

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

import brat_reader as br


class BratMultiTaskDataset(Dataset):
    EXAMPLE_TYPE = None
    ENCODINGS = {}
    SORTED_ATTRIBUTES = []
    START_ENTITY_MARKER = '@'
    END_ENTITY_MARKER = '@'

    def __init__(self, data_dir, sentences_dir,
                 label_names="all", window_size=0,
                 max_examples=-1, mark_entities=False):
        self.data_dir = data_dir
        self.sentences_dir = sentences_dir
        self.label_names = label_names
        self.window_size = window_size
        self.max_examples = max_examples
        self.mark_entities = mark_entities

        self.examples, self.docids_to_texts = self._get_examples_and_texts()
        if label_names == "all":
            self.label_names = self.SORTED_ATTRIBUTES
        else:
            for name in self.label_names:
                if name not in self.SORTED_ATTRIBUTES:
                    raise ValueError(f"Unknown attribute {name}")
            self.label_names = sorted(label_names)

    def __len__(self):
        if self.max_examples == -1:
            return len(self.examples)
        else:
            return len(self.examples[:self.max_examples])

    def __getitem__(self, idx):
        # Get the sentences in the window
        example = self.examples[idx]
        sentences = self.docids_to_texts[example.docid]
        si = example.sent_index
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
        entity_start = example.start_index - context[0]["start_index"]
        entity_end = example.end_index - context[0]["start_index"]
        # Surround the entity spans with '@'. E.g., 'He took @Toradol@'.
        if self.mark_entities is True:
            marked_text = text[:entity_start]
            marked_text += (self.START_ENTITY_MARKER +
                            text[entity_start:entity_end] +
                            self.END_ENTITY_MARKER)
            marked_text += text[entity_end:]
            text = marked_text
            entity_end += 2

        # Encode the labels
        labels = {}
        for attr_name in self.SORTED_ATTRIBUTES:
            if attr_name in self.label_names:
                if isinstance(example, br.Event):
                    try:
                        val = example.attributes[attr_name].value
                    except KeyError as e:
                        print(e)
                        print(example)
                        input()
                elif isinstance(example, br.Attribute):
                    if example.type != attr_name:
                        continue
                    val = example.value
                else:
                    raise ValueError(f"Found unsupported example type '{type(example)}'.")  # noqa
                labels[attr_name] = self.ENCODINGS[attr_name][val]

        return {"text": text,
                "entity_span": (entity_start, entity_end),
                # For reconstructing original entity span offsets
                #  in the source document.
                "char_offset": context[0]["start_index"],
                "labels": labels,
                "docid": example.docid}

    def preprocess_example(self, example):
        """
        Override with your own preprocessing logic.
        """
        return example

    def _get_examples_and_texts(self):
        """
        Pair each disposition event with a sentence
        in a document. Adds the "docid" and "sent_index"
        fields to the disposition.
        """
        all_examples = []
        docids_to_texts = {}
        ann_glob = os.path.join(self.data_dir, "*.ann")
        ann_files = glob(ann_glob)
        if ann_files == []:
            raise OSError(f"No annotations found at {ann_glob}")
        for ann_file in glob(ann_glob):
            anns = br.BratAnnotations.from_file(ann_file)
            examples = anns.get_highest_level_annotations(
                    type=self.EXAMPLE_TYPE)
            examples = [self.preprocess_example(ex) for ex in examples]
            docid = os.path.splitext(os.path.basename(ann_file))[0]
            for ex in examples:
                ex.update("docid", docid)
            all_examples.extend(examples)

            txt_file = os.path.join(self.sentences_dir, f"{docid}.txt.json")
            with open(txt_file, 'r') as inF:
                sent_data = [json.loads(line) for line in inF]
            # list of indices in sent_data, one per disposition.
            sent_idxs = self._match_examples_to_sentences(
                    examples, sent_data)
            for (ex, si) in zip(examples, sent_idxs):
                ex.update("sent_index", si)
            docids_to_texts[docid] = sent_data
        return all_examples, docids_to_texts

    def _match_examples_to_sentences(self, examples, sentences):
        # Each character index in the full document maps to an
        #  index in sentences.
        sent_index_lookup = {}
        for (i, sent) in enumerate(sentences):
            for j in range(sent["start_index"], sent["end_index"]):
                sent_index_lookup[j] = i

        sent_idxs = []
        for ex in examples:
            try:
                sent_i = sent_index_lookup[ex.start_index]
            except KeyError:
                try:
                    sent_i = sent_index_lookup[ex.end_index]
                except KeyError:
                    print(f"MISSING {ex}")
                    continue
            sent_idxs.append(sent_i)
        return sent_idxs

    def inverse_transform(self, task=None, encoded_labels=None):
        if task is None and encoded_labels is None:
            raise ValueError("must specify task and encoded labels, or dict of {task: encoded_labels}")  # noqa
        if getattr(self, "_inverse_encodings", None) is None:
            self._inverse_encodings = self._invert_encodings()
        if task is None:
            if isinstance(encoded_labels, dict):
                encoded = {}
                for (task, encs) in encoded_labels.items():
                    encoded[task] = self.inverse_transform(task, encs)
                return encoded
            else:
                raise ValueError("if task is None, encoded_labels must be a dict of {task: encoded_labels}")  # noqa
        if isinstance(encoded_labels, int):
            return self._inverse_encodings[task][encoded_labels]
        elif torch.is_tensor(encoded_labels):
            if encoded_labels.dim() == 0:
                return self._inverse_encodings[task][encoded_labels.item()]
            else:
                return [self._inverse_encodings[task][enc_lab.item()]
                        for enc_lab in encoded_labels]
        else:
            return [self._inverse_encodings[task][enc_lab]
                    for enc_lab in encoded_labels]

    def _invert_encodings(self):
        inv_enc = {}
        for (task, label_encs) in self.ENCODINGS.items():
            inv_enc[task] = {}
            for (label, enc) in label_encs.items():
                inv_enc[task][enc] = label
        return inv_enc


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
