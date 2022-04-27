import os
import argparse
import warnings
from glob import glob

import spacy  # noqa
import medspacy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=str,
                        help="Directory containing .txt files.")
    parser.add_argument("outdir", type=str,
                        help="Where to save ner annotations.")
    parser.add_argument("--model", type=str, choices=["transformer", "vector"],
                        help="Use the RoBERTa model or the vector model.")
    return parser.parse_args()


def main(args):
    glob_path = os.path.join(args.indir, "*.txt")
    glob_files = glob(glob_path)

    os.makedirs(args.outdir, exist_ok=False)

    if args.model.lower() == "transformer":
        model_name = "en_core_med7_trf"
    elif args.model.lower() == "vector":
        model_name = "en_core_med7_lg"
    else:
        raise ValueError(f"Unsupported model '{args.model}'")
    nlp = medspacy.load(model_name)
    for fpath in tqdm(glob_files):
        with open(fpath, 'r') as inF:
            txt = inF.read()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            doc = nlp(txt)
        med_ents = [e for e in doc.ents if e.label_ == "DRUG"]
        spans = [(e.start_char, e.end_char, e.text) for e in med_ents]
        formatted = format_as_brat(spans)
        bn = os.path.basename(fpath)
        outfile = f"{bn}.ner.ann"
        outpath = os.path.join(args.outdir, outfile)
        with open(outpath, 'w') as outF:
            for line in formatted:
                outF.write(line + '\n')


def format_as_brat(spans):
    formatted = []
    for (i, span) in enumerate(spans):
        text = span[2].replace('\n', ' ')
        line = f"T{i}\tNERPrediction {span[0]} {span[1]}\t{text}"
        formatted.append(line)
    return formatted


if __name__ == "__main__":
    args = parse_args()
    main(args)
