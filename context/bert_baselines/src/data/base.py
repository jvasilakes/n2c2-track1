import os
import json
import string
import warnings
from glob import glob
from collections import defaultdict
from typing import List, Dict, Union

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import brat_reader as br
from src.data.word_filters import POSFilter


class BratMultiTaskDataset(Dataset):
    EXAMPLE_TYPE = None
    ENCODINGS = {}

    def __init__(self, data_dir, sentences_dir,
                 label_names="all", load_labels=True, window_size=0,
                 max_examples=-1, mark_entities=False, entity_markers=None):
        self.data_dir = data_dir
        self.sentences_dir = sentences_dir
        self.label_names = self._validate_label_names(label_names)
        self.load_labels = load_labels
        self.window_size = window_size
        self.max_examples = max_examples
        self.mark_entities = mark_entities
        self.entity_markers = self._validate_entity_markers(entity_markers)
        self._name = None

        self.examples, self.docids_to_texts = self._get_examples_and_texts()

    def _validate_label_names(self, label_names):
        if label_names == "all":
            label_names = sorted(list(self.ENCODINGS.keys()))
        else:
            _tmp_label_names = []
            for name in label_names:
                if name not in self.ENCODINGS.keys():
                    raise ValueError(f"Unknown attribute {name}")
                _tmp_label_names.append(name)
            label_names = sorted(_tmp_label_names)
        return label_names

    def _validate_entity_markers(self, entity_markers):
        if entity_markers is None:
            return entity_markers
        assert isinstance(entity_markers, (list, tuple)), "entity_markers not list or tuple"  # noqa
        assert len(entity_markers) == 2, "len(entity_markers) != 2"
        msg = "not all entity_markers are str"
        assert all([isinstance(marker, str) for marker in entity_markers]), msg
        return entity_markers

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
        # An example is brat_reader.Event or Attribute
        example = self.examples[idx]
        sentences = self.docids_to_texts[example.docid]
        # Get the sentences in the window
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

        # Compute the relative character offset of the entity in the context
        entity_start = example.start_index - context[0]["start_index"]
        entity_end = example.end_index - context[0]["start_index"]
        # Surround the entity spans with the specified markers.
        #  E.g., self.entity_markers = '@', 'He took @Toradol@'.
        if self.entity_markers is not None:
            marked_text = text[:entity_start]
            marked_text += (self.entity_markers[0] +
                            text[entity_start:entity_end] +
                            self.entity_markers[1])
            marked_text += text[entity_end:]
            text = marked_text
            entity_end += sum([len(marker) for marker in self.entity_markers])

        # Encode the labels
        labels = {}
        if self.load_labels is True:
            for attr_name in self.label_names:
                if isinstance(example, br.Event):
                    try:
                        val = example.attributes[attr_name].value
                    except KeyError:
                        raise KeyError(f"Didn't find attribute {attr_name} in {example.attributes}")  # noqa
                elif isinstance(example, br.Attribute):
                    if example.type != attr_name:
                        continue
                    val = example.value
                else:
                    raise ValueError(f"Found unsupported example type '{type(example)}'.")  # noqa
                labels[attr_name] = self.ENCODINGS[attr_name][val]

        return {"text": text,
                "entity_char_span": (entity_start, entity_end),
                # For reconstructing original entity span offsets
                #  in the source document.
                "char_offset": context[0]["start_index"],
                "labels": labels or None,
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

    levitate_window_size: number of spans to consider +/- the target entity
    levitated_pos_tags: list of UPOS tags to specify which levitated words
                        to use. E.g., levitated_pos_tags=["AUX", "VERB"]
                        will keep all verbs and auxiliaries. If None, does
                        not filter words.
    """
    def __init__(
            self,
            bert_model_name_or_path,
            max_seq_length=128,
            tasks_to_load="all",
            mark_entities=False,
            entity_markers=None,
            use_levitated_markers=False,
            levitate_window_size=5,
            levitated_pos_tags=None,
            name=None):
        self.bert_model_name_or_path = bert_model_name_or_path
        self.max_seq_length = max_seq_length

        self.mark_entities = mark_entities
        if self.mark_entities is True:
            msg = "mark_entities is deprecated. Using entity_markers instead."
            self.mark_entities = False
            if entity_markers is None:
                entity_markers = '@'
                msg += " Defaulting to '@' entity markers."
            warnings.warn(msg, DeprecationWarning)
        self.entity_markers = self._validate_entity_markers(entity_markers)

        self.use_levitated_markers = use_levitated_markers
        self.levitate_window_size = levitate_window_size
        self.levitated_pos_tags = levitated_pos_tags
        self._name = name

        self.tokenizer = AutoTokenizer.from_pretrained(
                self.bert_model_name_or_path, use_fast=True)

        if self.levitated_pos_tags is not None:
            self.word_filter = POSFilter("en_core_sci_scibert", self.tokenizer,
                                         keep_tags=self.levitated_pos_tags)
        else:
            self.word_filter = None

    def _validate_entity_markers(self, entity_markers):
        if entity_markers is None:
            return entity_markers
        elif isinstance(entity_markers, str):
            return (entity_markers, entity_markers)
        elif isinstance(entity_markers, (list, tuple)):
            assert all([isinstance(marker, str) for marker in entity_markers]), "Entity markers must be all strings!"  # noqa
            if len(entity_markers) == 1:
                return (entity_markers[0], entity_markers[0])
            elif len(entity_markers) == 2:
                return entity_markers
            else:
                raise ValueError(f"Incorrect number of entity markers speciied {len(self.entity_markers)}. Should be length 1 or 2, or None.")  # noqa

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
        batch = {
            "encodings": None,  # output of BertTokenizer
            "entity_token_idxs": [],  # indices of entity's wordpiece tokens
            "entity_char_spans": [],  # (start,end) character indices of entity
            "texts": [],  # the original text that was transformed into encodings.  # noqa
            "char_offsets": [], # start character index of the texts in the documents.  # noqa
            "labels": defaultdict(list),  # dict of task->labels
            "docids": []  # source documents for examples in this batch
            # Optional: levitated_marker_idxs: []
        }

        # Collate entity_char_spans, char_offsets, texts, docids
        for ex in examples:
            batch["entity_char_spans"].append(ex["entity_char_span"])
            batch["char_offsets"].append(ex["char_offset"])
            batch["texts"].append(ex["text"])
            batch["docids"].append(ex["docid"])
        batch["entity_char_spans"] = torch.tensor(batch["entity_char_spans"])

        # Collate labels
        # batch["labels"] == None if all example["labels"] == None
        for ex in examples:
            if ex["labels"] is None:
                batch["labels"] = None
            else:
                for (task, val) in ex["labels"].items():
                    if batch["labels"] is None:
                        msg = "Found some examples with labels and some without!"  # noqa
                        raise ValueError(msg)
                    batch["labels"][task].append(val)
        if batch["labels"] is not None:
            for task in batch["labels"].keys():
                batch["labels"][task] = torch.tensor(batch["labels"][task])
            # convert defaultdict to regular dict
            batch["labels"] = dict(batch["labels"])

        # Get encodings
        encodings = self.tokenizer(batch["texts"], truncation=True,
                                   max_length=self.max_seq_length,
                                   padding="max_length",
                                   return_offsets_mapping=True,
                                   return_tensors="pt")

        # Populate entity_token_idxs
        for (example_i, entity_span) in enumerate(batch["entity_char_spans"]):
            entity_token_idxs = self.get_token_indices_from_char_span(
                    entity_span, encodings["offset_mapping"][example_i])
            if len(entity_token_idxs) == 0:
                raise ValueError(f"Couldn't find entity span {entity_span} in document {examples[example_i]['docid']}!")  # noqa
            batch["entity_token_idxs"].append(entity_token_idxs)

        # Optional: Packed-Levitated Marker modifies the encodings.
        if self.use_levitated_markers is True:
            tmp = self.levitate_encodings(
                encodings, batch["entity_token_idxs"])
            encodings, batch["levitated_marker_idxs"] = tmp

        # Populate encodings, which will have been modified
        #  if use_levitated_markers=True.
        batch["encodings"] = encodings
        return batch

    def get_token_indices_from_char_span(self, char_span, offset_mapping):
        token_idxs = []
        for (i, offset) in enumerate(offset_mapping):
            # zero-length span, usually [0, 0] for [PAD]
            if offset[0] == offset[1]:
                continue
            if offset[0] >= char_span[0] and offset[1] <= char_span[1]:
                token_idxs.append(i)
        return token_idxs

    def levitate_encodings(self, encodings, entity_token_idxs):
        """
        encodings: output of self.tokenizer(texts)
        entity_token_idxs: list of token indices corresponding to the entity
        """
        new_encodings = {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "attention_mask": [],
            "offset_mapping": [],
        }
        batch_size, max_seq_length = encodings["input_ids"].size()
        # There are up to self.levitate_window_size spans on each
        # side of the entity, and each span has a start and end marker,
        # so the max number of markers is the window size times 4.
        max_num_markers = self.levitate_window_size * 4
        max_levitated_seq_length = max_seq_length + max_num_markers

        # Extend the token_type_ids to cover the markers.
        new_encodings["token_type_ids"] = torch.hstack(
            (encodings["token_type_ids"],
             torch.zeros(batch_size, max_num_markers, dtype=torch.long))
        )
        # Extend the offset_mapping to cover the markers.
        new_encodings["offset_mapping"] = torch.hstack(
            (encodings["offset_mapping"],
             torch.zeros(batch_size, max_num_markers, 2, dtype=torch.long))
        )
        # Find and add the levitated markers.
        levitated_marker_idxs = [[] for _ in range(batch_size)]
        for example_idx in range(batch_size):
            input_ids = encodings["input_ids"][example_idx]
            attention_mask = encodings["attention_mask"][example_idx]

            # attention_mask is a square matrix with x and y dimensions
            #  equal to max_levitated_seq_length.
            attention_mask = torch.hstack((attention_mask,
                                           torch.zeros(max_num_markers)))
            attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.repeat(max_levitated_seq_length, 1)
            position_ids = torch.arange(max_levitated_seq_length,
                                        dtype=torch.long)
            input_ids = torch.cat(
                (input_ids, torch.zeros(max_num_markers, dtype=torch.long)))

            # Make sure we don't use the same tokens as the entity
            entity_marker_ids = self.tokenizer.convert_tokens_to_ids(
                    self.entity_markers)
            start_token_id = 1
            end_token_id = 2  # actual token is "[unused{token_id}]"
            while start_token_id not in entity_marker_ids and end_token_id not in entity_marker_ids:  # noqa
                start_token_id += 1
                end_token_id += 1

            # Get all tokens before and after the entity mention,
            # up to self.levitate_window_size.
            lev_span_idxs = self.get_levitated_spans(
                input_ids, entity_token_idxs[example_idx],
                word_filter=self.word_filter)
            #if len(lev_span_idxs) == 0:
            #    seq_len_no_pad = (input_ids != 0).sum()
            #    input_text = self.tokenizer.decode(input_ids[:seq_len_no_pad])
            #    warnings.warn(f"No levitated spans found for '{input_text}'")

            # Each levitated marker has a start and end token,
            # so we iterate by 2 indices starting from the end of the input.
            marker_start_idxs = range(max_seq_length,
                                      max_levitated_seq_length, 2)
            for (marker_start, span) in zip(marker_start_idxs, lev_span_idxs):
                # The indices of the markers in position_ids and attention_mask
                marker_end = marker_start + 1
                levitated_marker_idxs[example_idx].extend(
                    [marker_start, marker_end])
                # position ids of the markers to equal those of the start/end
                # tokens of the corresponding span.
                position_ids[marker_start] = span[0]
                position_ids[marker_end] = span[-1]
                # Markers are visible to each other only, and can see the text
                attention_mask[marker_start, [marker_start, marker_end]] = 1
                attention_mask[marker_end, [marker_start, marker_end]] = 1
                input_ids[marker_start] = start_token_id
                input_ids[marker_end] = end_token_id

            new_encodings["input_ids"].append(input_ids)
            new_encodings["position_ids"].append(position_ids)
            new_encodings["attention_mask"].append(attention_mask.unsqueeze(0))
        new_encodings["input_ids"] = torch.vstack(new_encodings["input_ids"])
        new_encodings["position_ids"] = torch.vstack(
            new_encodings["position_ids"])
        new_encodings["attention_mask"] = torch.vstack(
            new_encodings["attention_mask"])
        return new_encodings, levitated_marker_idxs

    def get_levitated_spans(self, input_ids, entity_token_idxs,
                            word_filter=None):
        """
        Gets word spans (collapsing wordpiece tokens) within a
        window of the entity mention in the input.
        Returns wordpiece token indices for each word span.
        """
        entity_start = entity_token_idxs[0]
        entity_end = entity_token_idxs[-1]

        # Get all the tokens before and after the entity mention
        before_token_idxs = np.arange(0, entity_start)
        seq_len_no_pad = (input_ids != 0).sum()
        after_token_idxs = np.arange(entity_end+1, seq_len_no_pad)

        # Get token spans across wordpiece boundaries
        before_token_spans = self.collapse_wordpiece(
            before_token_idxs, input_ids, rm_special_tokens=True)
        after_token_spans = self.collapse_wordpiece(
            after_token_idxs, input_ids, rm_special_tokens=True)

        # Optionally filter spans using a word_filter class instance
        if word_filter is not None:
            before_token_spans = word_filter(before_token_spans, input_ids)
            after_token_spans = word_filter(after_token_spans, input_ids)

        # Keep only the spans in the window
        before_token_spans = before_token_spans[-self.levitate_window_size:]
        after_token_spans = after_token_spans[:self.levitate_window_size]
        return [*before_token_spans, *after_token_spans]

    def collapse_wordpiece(self, token_idxs, input_ids,
                           rm_special_tokens=True):
        special_tokens = list(self.tokenizer.special_tokens_map.values())
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        out_token_spans = []
        current_token_span = []
        for token_idx in token_idxs:
            token = tokens[token_idx]
            if rm_special_tokens is True:
                if token in special_tokens:
                    continue
            if token.startswith("##"):
                current_token_span.append(token_idx)
            else:
                if current_token_span != []:
                    out_token_spans.append(current_token_span)
                current_token_span = [token_idx]
        if current_token_span != []:
            out_token_spans.append(current_token_span)
        return out_token_spans

    def get_subspan_idxs(self, idxs, input_ids,
                         max_length=2, ignore_punct=True):
        """
        Use input_ids to check if we should ignore the token at
        a given index.
        """
        ignore_token_ids = set()
        if ignore_punct is True:
            ignore_tokens = list(string.punctuation)
            ignore_token_ids = self.tokenizer.convert_tokens_to_ids(ignore_tokens)  # noqa
            ignore_token_ids = set(ignore_token_ids)
        if torch.is_tensor(idxs):
            idxs = idxs.tolist()
        if isinstance(idxs, np.ndarray):
            idxs = list(idxs)
        subspans = []
        for j in range(1, len(idxs) + 1):
            start = max(0, j - max_length)
            for i in range(start, j):
                subspan = idxs[i:j]
                subspan_tokens = input_ids[subspan].tolist()
                if len(ignore_token_ids.intersection(subspan_tokens)) == 0:
                    subspans.append(subspan)
        return subspans
