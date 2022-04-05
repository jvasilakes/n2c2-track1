import os
import json
from glob import glob
from collections import defaultdict
from typing import List, Dict, Union

import torch
import numpy as np
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
                 max_examples=-1, mark_entities=False,
                 levitate=False):
        self.data_dir = data_dir
        self.sentences_dir = sentences_dir
        self.label_names = label_names
        self.window_size = window_size
        self.max_examples = max_examples
        self.mark_entities = mark_entities
        self.levitate = levitate
        self._name = None

        self.examples, self.docids_to_texts = self._get_examples_and_texts()
        if label_names == "all":
            self.label_names = self.SORTED_ATTRIBUTES
        else:
            for name in self.label_names:
                if name not in self.SORTED_ATTRIBUTES:
                    raise ValueError(f"Unknown attribute {name}")
            self.label_names = sorted(label_names)

    @property
    def name(self) -> str:
        if self._name is None:
            cls_str = str(self.__class__)
            cls_name = cls_str.split('.')[-1]
            self._name = cls_name.replace("'>", '')
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

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

    def preprocess_example(self, example: br.Annotation):
        """
        Override with your own preprocessing logic.
        """
        return example

    def filter_examples(self, examples: List[br.Annotation]):
        """
        Override with your own preprocessing logic.
        """
        return examples

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
            examples = self.filter_examples(
                    [self.preprocess_example(ex) for ex in examples])
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

        # Packed-Levitated Marker
        if self.levitate is True:
            encodings = self.levitate_encodings(
                encodings, batch["entity_spans"],
                window_size=self.num_levitated_tokens)

        batch["encodings"] = encodings
        batch["entity_spans"] = torch.tensor(batch["entity_spans"])
        for task in batch["labels"].keys():
            batch["labels"][task] = torch.tensor(batch["labels"][task])
        batch["labels"] = dict(batch["labels"])
        return batch

    def levitate_encodings(self, encodings, entity_spans,
                           window_size=5, max_span_length=3):
        """
        encodings: output of self.tokenizer(texts)
        entity_spans: list of (start_index, end_index) tuples.
        window_size: number of spans to consider +/- the target entity
        max_span_length: maximum length in number of wordpiece tokens
                         to treat as a marked span.
        """
        for example_idx in range(entity_spans.size(0)):
            input_ids = encodings["input_ids"][example_idx]
            attention_mask = encodings["attention_mask"][example_idx]
            offsets = encodings["offset_mapping"][example_idx]
            entity_span = entity_spans[example_idx]

            entity_token_idxs = [i for (i, off) in enumerate(offsets)
                                 if off[0] >= entity_span[0]
                                 and off[1] <= entity_span[1]]
            if len(entity_token_idxs) == 0:
                raise ValueError("Couldn't find entity span!")
            entity_start = entity_token_idxs[0]
            entity_end = entity_token_idxs[1]

            # Get tokens in window_size around start/end entity_token_idxs
            idx_before_entity = max(0, entity_token_idxs[0] - window_size)
            seq_len_no_pad = (input_ids != 0).sum()
            idx_after_entity = min(seq_len_no_pad,
                                   entity_token_idxs[-1] + window_size)

            before_token_idxs = np.arange(idx_before_entity, entity_start)
            before_span_idxs = self.get_subspan_idxs(
                before_token_idxs, input_ids, max_length=max_span_length)
            after_token_idxs = np.arange(entity_end+1, idx_after_entity)
            after_span_idxs = self.get_subspan_idxs(
                after_token_idxs, input_ids, max_length=max_span_length)

            # max_num_spans is the number of sublists up to length max_length
            # E.g., for window_size 5 and max_length 3 there are 5+4+3=12
            #  possible subspans on each side (12*2=24) of the entity span.
            max_num_spans = np.arange(window_size+1)[-max_span_length:].sum() * 2  # noqa
            # Each span has a start and end marker, so multiply by 2 again.
            max_num_markers = max_num_spans * 2
            max_seq_length = len(input_ids)
            max_levitated_seq_length = max_seq_length + max_num_markers

            position_ids = torch.arange(max_levitated_seq_length,
                                        dtype=torch.long)
            # attention_mask should be a square matrix with x and y dimentions
            #  equal to (max_seq_length + max_num_markers)
            attention_mask = torch.hstack((attention_mask,
                                           torch.zeros(max_num_markers)))
            attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.repeat(max_levitated_seq_length, 1)

            # TODO: what should position_ids be for PAD tokens?
            all_span_idxs = [*before_span_idxs, *after_span_idxs]
            marker_start_idxs = range(max_seq_length + 1,
                                      max_levitated_seq_length + 1, 2)
            for (marker_start, span) in zip(marker_start_idxs, all_span_idxs):
                # The indices of the markers in position_ids and attention_mask
                marker_end = marker_start + 1
                # position ids of the markers to equal those of the start/end
                # tokens of the span.
                position_ids[marker_start] = span[0]
                position_ids[marker_end] = span[-1]
                # Markers are visible to each other and other markers.
                attention_mask[marker_start, marker_start] = 1
                attention_mask[marker_start, marker_end] = 1
                attention_mask[marker_end, marker_end] = 1
                attention_mask[marker_end, marker_start] = 1

    def get_subspan_idxs(self, idxs, input_ids, max_length=2, ignore_tokens=[]):  # noqa
        """
        Use input_ids to check if we should ignore the token at
        a given index.
        """
        if torch.is_tensor(idxs):
            idxs = idxs.tolist()
        subspans = []
        for j in range(1, len(idxs) + 1):
            start = max(0, j - max_length)
            for i in range(start, j):
                subspan = idxs[i:j]
                subspan_tokens = input_ids[subspan].tolist()
                if len(ignore_tokens.intersection(subspan_tokens)) == 0:
                    subspans.append(subspan)
        return subspans
