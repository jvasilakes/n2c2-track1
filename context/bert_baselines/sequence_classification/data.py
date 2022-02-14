import os
import json
import warnings
from glob import glob

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from brat_reader import BratAnnotations


class n2c2RawTextDataset(Dataset):
    def __init__(self, data_dir):
        raise NotImplementedError()
        self.data_dir = data_dir
        self.events, self.docids_to_texts = self._get_dispositions_and_texts()

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event = self.events[idx]
        return (event, self.docids_to_texts[event.docid])

    def _get_dispositions_and_texts(self):
        all_dispositions = []
        docids_to_texts = {}
        ann_glob = os.path.join(self.data_dir, "*.ann")
        for ann_file in glob(ann_glob):
            anns = BratAnnotations.from_file(ann_file)
            dispositions = anns.get_events_by_type("Disposition")
            docid = os.path.basename(ann_file).strip(".ann")
            for d in dispositions:
                d.update("docid", docid)
            all_dispositions.extend(dispositions)

            txt_file = os.path.join(self.data_dir, f"{docid}.txt")
            with open(txt_file, 'r') as inF:
                text = inF.read()
            docids_to_texts[docid] = text
        return all_dispositions, docids_to_texts


class n2c2SentencesDataset(Dataset):
    def __init__(self, data_dir, sentences_dir, window_size=0):
        self.data_dir = data_dir
        self.sentences_dir = sentences_dir
        self.window_size = window_size
        self.events, self.docids_to_texts = self._get_dispositions_and_texts()

    def __len__(self):
        return len(self.events)

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

        return {"text": text,
                "entity_span": (entity_start, entity_end),
                "docid": event.docid}

    def _get_dispositions_and_texts(self):
        all_dispositions = []
        docids_to_texts = {}
        ann_glob = os.path.join(self.data_dir, "*.ann")
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


class n2c2SentencesDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, sentences_dir, batch_size,
                 model_name_or_path, max_seq_length=128, window_size=0):
        super().__init__()
        self.data_dir = data_dir
        self.sentences_dir = sentences_dir
        self.batch_size = batch_size
        self.model_name_or_path = model_name_or_path
        self.window_size = window_size

        self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, use_fast=True)

    def setup(self, stage=None):
        train_path = os.path.join(self.data_dir, "train")
        train_sent_path = os.path.join(self.sentences_dir, "train")
        self.train = n2c2SentencesDataset(
                train_path, train_sent_path, window_size=self.window_size)

        val_path = os.path.join(self.data_dir, "dev")
        val_sent_path = os.path.join(self.sentences_dir, "train")
        self.val = n2c2SentencesDataset(
                val_path, val_sent_path, window_size=self.window_size)

        test_path = os.path.join(self.data_dir, "test")
        test_sent_path = os.path.join(self.sentences_dir, "train")
        if os.path.exists(test_path):
            self.test = n2c2SentencesDataset(
                    test_path, test_sent_path, window_size=self.window_size)
        else:
            warnings.warn("No test set found.")
            self.test = None

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        if self.test is not None:
            return DataLoader(self.test, batch_size=self.batch_size)
        return None

    def encode(self, example):
        if self.window_size == 0:
            return example
        else:
            raise NotImplementedError()
