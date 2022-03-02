import os
import json
import argparse
from hashlib import md5
from itertools import zip_longest

import spacy
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help="Path to dSentences.npz")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to save the data.")
    parser.add_argument("--train_size", type=float, default=0.75,
                        help="Ratio of the full dataset to allocate to train.")
    return parser.parse_args()


def main(args):
    dsents = np.load(args.infile, encoding="latin1", allow_pickle=True)
    latent_names = ["verb_obj_tuple", "obj_sing_pl", "gender",
                    "subj_sing_pl", "sent_type", "nr_person",
                    "pos_neg_verb", "verb_tense", "verb_style"]

    nlp = spacy.load("en_core_web_trf")
    outdata = []
    sents = [sent.decode("utf-8") for sent in dsents["sentences_array"]]
    pbar = tqdm(total=len(sents))
    for (i, doc) in enumerate(nlp.pipe(sents, disable=["ner"], n_process=4)):
        sent_toks = []
        sent_pos = []
        for tok in doc:
            sent_toks.append(str(tok))
            sent_pos.append(str(tok.pos_))

        classes = dict(zip(latent_names, dsents["latents_classes"][i]))
        classes = {k: int(v) for (k, v) in classes.items()}
        outdata.append(
                {"uid": md5(sents[i].encode()).hexdigest(),
                 "tokens": sent_toks,
                 "pos": sent_pos,
                 "labels": classes
                 }
                )
        pbar.update(1)
        if i == 2199:
            break

    metadata = dsents["metadata"][()]
    train, dev = split_by_content(outdata, metadata,
                                  train_size=args.train_size)
    traindir = os.path.join(args.outdir, "train")
    os.makedirs(traindir, exist_ok=False)
    save_dataset(train, traindir, dataset_name="train")
    devdir = os.path.join(args.outdir, "dev")
    os.makedirs(devdir, exist_ok=False)
    save_dataset(dev, devdir, dataset_name="dev")


def split_by_content(data, metadata, train_size=0.75):
    content_size = metadata["latent_sizes"][0]

    train = []
    dev = []
    for (i, chunk) in enumerate(grouper(data, content_size)):
        train_chunk, dev_chunk = train_test_split(
                chunk, random_state=i, train_size=train_size)
        train.extend(train_chunk)
        dev.extend(dev_chunk)
    return train, dev


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=None, *args)


def save_dataset(dataset, outdir, dataset_name="dataset"):
    outpath = os.path.join(outdir, f"{dataset_name}.jsonl")
    with open(outpath, 'w') as outF:
        for ex in dataset:
            json.dump(ex, outF)
            outF.write('\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)
