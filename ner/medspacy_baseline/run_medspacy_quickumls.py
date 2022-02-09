import os
import argparse
import warnings
from glob import glob

import spacy  # noqa
import medspacy
from medspacy.util import DEFAULT_PIPENAMES
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=str,
                        help="Directory containing .txt files.")
    parser.add_argument("outdir", type=str,
                        help="Where to save ner annotations.")
    return parser.parse_args()


def main(args):
    glob_path = os.path.join(args.indir, "*.txt")
    glob_files = glob(glob_path)

    os.makedirs(args.outdir, exist_ok=False)

    medspacy_pipes = DEFAULT_PIPENAMES.copy()
    if "medspacy_quickumls" not in medspacy_pipes:
        medspacy_pipes.add("medspacy_quickumls")

    # Chemicals and Drugs Semantic Group. See
    #  https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemGroups_2018.txt
    rx_tuis = set(["T116", "T195", "T123", "T122", "T103", "T120", "T104",
                   "T200", "T196", "T126", "T131", "T125", "T129", "T130",
                   "T197", "T114", "T109", "T121", "T192", "T127"])

    nlp = medspacy.load(enable=medspacy_pipes)
    for fpath in tqdm(glob_files):
        with open(fpath, 'r') as inF:
            txt = inF.read()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            doc = nlp(txt)
        med_ents = [e for e in doc.ents
                    if len(e._.semtypes.intersection(rx_tuis)) > 0]
        # Get the set since we only care about unique spans, not CUIs.
        spans = set([(e.start_char, e.end_char, e.text) for e in med_ents])
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
