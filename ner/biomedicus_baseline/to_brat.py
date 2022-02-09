import os
import json
import argparse
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=str,
                        help="""Directory containing json formatted
                                biomedicus output files.""")
    parser.add_argument("outdir", type=str,
                        help="Where to save the formatted output")
    parser.add_argument("--annotation-type", type=str, choices=["ner"])
    return parser.parse_args()


def main(args):
    glob_path = os.path.join(args.indir, "*.json")
    glob_files = glob(glob_path)

    format_fn = FORMAT_FNS[args.annotation_type]

    os.makedirs(args.outdir, exist_ok=False)
    for fname in glob_files:
        data = json.load(open(fname))
        formatted = format_fn(data)
        outpath = os.path.join(args.outdir, os.path.basename(fname))
        outpath = outpath.replace(".json", f".{args.annotation_type}.ann")
        with open(outpath, 'w') as outF:
            for line in formatted:
                outF.write(line + '\n')


def format_ner(json_data, field="umls_concepts"):
    formatted = []
    # Chemicals and Drugs Semantic Group. See
    #  https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemGroups_2018.txt
    rx_tuis = ["T116", "T195", "T123", "T122", "T103", "T120", "T104", "T200",
               "T196", "T126", "T131", "T125", "T129", "T130", "T197", "T114",
               "T109", "T121", "T192", "T127"]
    preds = json_data["documents"]["plaintext"]["label_indices"][field]["labels"]  # noqa
    seen_spans = set()
    for pred in preds:
        if pred["fields"]["tui"] not in rx_tuis:
            continue
        start_i = pred["start_index"]
        end_i = pred["end_index"]
        if (start_i, end_i) in seen_spans:
            continue
        seen_spans.add((start_i, end_i))
        text = pred["_text"].replace('\n', ' ')
        i = len(seen_spans)
        line = f"T{i}\tNERPrediction {start_i} {end_i}\t{text}"
        formatted.append(line)
    return formatted


FORMAT_FNS = {"ner": format_ner}


if __name__ == "__main__":
    args = parse_args()
    main(args)
