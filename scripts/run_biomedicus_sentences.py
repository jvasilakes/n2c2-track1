import os
import json
import yaml
import random
import argparse
from yaml import Loader
from glob import glob
from tqdm import tqdm

import torch
import numpy as np

from biomedicus.sentences.input import InputMapping
from biomedicus.sentences.vocabulary import n_chars
from biomedicus.sentences.bi_lstm import BiLSTM, load_vectors, \
                                         load_char_mapping, predict


"""
This script applies the clinical sentence segmentation model from
to a single file via --infile or a directory of .txt files via --indir

Usage:
    python run_biomedicus_sentences.py \
            --biomedicus_data_dir ~/.biomedicus/data/sentences/ \
            --infile /path/to/file.txt || --indir /path/to/dir \
            --outdir /path/to/output/dir

--infile and --indir are mutually exclusive arguments.
--outdir must not already exist

For each file the input, outputs a JSON lines file with one sentence per
line. The JSON format is

{"sent_index": int  # Index of the sentence in the source file.
 "start_index": int, "end_index": int  # Start/end char indices of the sentence
 "_text": str  # The sentence text.
 }
"""


class Hparams():
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--biomedicus_data_dir", type=str, required=True,
                        help="Path to .biomedicus/data/sentences")
    parser.add_argument("--infile", type=str, default=None,
                        help="""Path to a text file to process. Either this
                                or --indir are required arguments.""")
    parser.add_argument("--indir", type=str, default=None,
                        help="""Path to a directory containing .txt files to
                                process. Either this or --infile are
                                required arguments.""")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to save the output.")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main(args):
    set_seed(0)

    if args.infile is None and args.indir is None:
        raise argparse.ArgumentError(
                args.infile, "Must specify --infile or --indir")
    elif args.infile is not None and args.indir is not None:
        raise argparse.ArgumentError(
                args.infile, "Must specify one one of --infile or --indir")
    if args.infile is not None:
        filepaths = [args.infile]
    elif args.indir is not None:
        filepaths = glob(os.path.join(args.indir, "*.txt"))

    os.makedirs(args.outdir, exist_ok=False)

    config_file = os.path.join(args.biomedicus_data_dir, "1579713474.213175.yml")  # noqa
    model_pt_file = os.path.join(args.biomedicus_data_dir, "1579713474.213175.pt")  # noqa
    char_file = os.path.join(args.biomedicus_data_dir, "chars.txt")
    emb_file = os.path.join(args.biomedicus_data_dir, "mimic100.vec")

    with open(config_file) as inF:
        d = yaml.load(inF, Loader)
    hparams = Hparams()
    hparams.__dict__.update(d)

    char_mapping = load_char_mapping(char_file)
    words, vectors = load_vectors(emb_file)

    model = BiLSTM(hparams, n_chars(char_mapping), vectors)
    with open(model_pt_file, 'rb') as inF:
        state_dict = torch.load(inF)
        model.load_state_dict(state_dict)
    print(model)

    for (fpath, text) in tqdm(input_iterator(filepaths), total=len(filepaths)):
        input_mapper = InputMapping(char_mapping, words, hparams.word_length)
        sentences = segment(model, text, input_mapper)
        bn = os.path.basename(fpath)
        outfile = os.path.join(args.outdir, f"{bn}.json")
        with open(outfile, 'w') as outF:
            for sent in sentences:
                json.dump(sent, outF)
                outF.write('\n')


def input_iterator(fpaths):
    for fpath in fpaths:
        with open(fpath, 'r', encoding="utf-8") as inF:
            yield fpath, inF.read()


def segment(model, text, input_mapper):
    token_spans, char_ids, word_ids = input_mapper.transform_text(text)
    preds = predict(model, char_ids, word_ids, device="cpu")
    preds = preds[0]

    all_sentences = []
    curr_sent_span = None
    for (ts, p) in zip(token_spans, preds):

        segment_lab = int(p.item())
        if segment_lab == 1:
            if curr_sent_span is not None:
                start_i, end_i = curr_sent_span
                sent_data = {"sent_index": len(all_sentences),
                             "start_index": start_i, "end_index": end_i,
                             "_text": text[start_i:end_i]}
                all_sentences.append(sent_data)
            curr_sent_span = list(ts)
        else:
            if curr_sent_span is None:
                # We haven't actually encountered a start of sentence yet.
                continue
            else:
                curr_sent_span[1] = ts[1]

    start_i, end_i = curr_sent_span
    sent_data = {"sent_index": len(all_sentences),
                 "start_index": start_i, "end_index": end_i,
                 "_text": text[start_i:end_i]}
    all_sentences.append(sent_data)

    return all_sentences


if __name__ == "__main__":
    args = parse_args()
    main(args)
