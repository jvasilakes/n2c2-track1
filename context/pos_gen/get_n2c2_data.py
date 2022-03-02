import os
import re
import json
import argparse
import warnings
from glob import glob
from hashlib import md5

import spacy
import scispacy  # noqa
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, required=True,
                        help="Directory containing {train/dev/test}/*.txt files.")  # noqa
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to save the data.")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["gen", "sci"],
                        help="Domain of the input data.")
    return parser.parse_args()


def main(args):
    if args.domain == "gen":
        nlp = spacy.load("en_core_web_trf")
    elif args.domain == "sci":
        nlp = spacy.load("en_core_sci_scibert")
    else:
        raise NotImplementedError()

    os.makedirs(args.outdir, exist_ok=False)

    # datasets = ["train", "dev", "test"]
    datasets = ["dev", "test"]
    lens = []
    for dataset in datasets:
        print(dataset)
        dataset_dir = os.path.join(args.indir, dataset)
        glob_path = os.path.join(dataset_dir, "*.txt")
        fpaths = glob(glob_path)
        if len(fpaths) == 0:
            warnings.warn(f"No {dataset} dataset found at {glob_path}")
        outdata = []
        for fpath in tqdm(fpaths):
            txt = open(fpath, 'r').read()
            txt = clean_text(txt)
            doc = nlp(txt)
            for sent in doc.sents:
                sent_toks = []
                sent_pos = []
                lens.append(len(sent))
                for tok in sent:
                    sent_toks.append(str(tok))
                    sent_pos.append(str(tok.pos_))
                outdata.append(
                    {"uid": md5(str(sent).encode()).hexdigest(),
                     "tokens": sent_toks,
                     "pos": sent_pos,
                     "src_file": os.path.basename(fpath).strip(".txt")
                     }
                    )

        print(max(lens))
        print(min(lens))
        outpath = os.path.join(args.outdir, f"{dataset}.jsonl")
        with open(outpath, 'w') as outF:
            for line in outdata:
                json.dump(line, outF)
                outF.write('\n')


def clean_text(text):
    """
    Remove section boundaries and normalize whitespace.
    """
    text = re.sub(r'[\-~=_\*]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if __name__ == "__main__":
    args = parse_args()
    main(args)
