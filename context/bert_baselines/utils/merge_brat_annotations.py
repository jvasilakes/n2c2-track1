import os
import argparse
from glob import glob
from collections import defaultdict

import brat_reader as br


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dirs", type=str, nargs='+', required=True,
                        help="""Paths to directories containing
                                brat annotation files.""")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Where to save the merged files.")
    parser.add_argument("--conflicts", type=str, default="error",
                        choices=["error", "keep_earlier", "keep_later"],
                        help="""If there is an attribute conflict, how to
                                resolve it.
                                'error': raise an error.
                                'keep_earlier': use the attribute which is
                                seen first from the files in --pred_dirs.
                                'keep_later': use the attribute which is
                                seen last from the files in --pred_dirs.""")
    return parser.parse_args()


def main(args):
    # Collect all predictions together by filename
    fname_to_anns = defaultdict(list)
    for pd in args.pred_dirs:
        ann_glob = os.path.join(pd, "*.ann")
        for ann_file in glob(ann_glob):
            bn = os.path.basename(ann_file)
            anns = br.BratAnnotations.from_file(ann_file)
            fname_to_anns[bn].append(anns)

    # Merged the collected predictions
    os.makedirs(args.outdir, exist_ok=False)
    for (fname, all_anns) in fname_to_anns.items():
        merged_anns = merge_brat(all_anns)
        bn = os.path.basename(fname)
        merged_anns.save_brat(args.outdir, bn)


def merge_brat(anns: list, conflicts="error"):
    if conflicts == "keep_earlier":
        conflicts = "keep_this"
    elif conflicts == "keep_later":
        conflicts = "keep other"
    merged_anns = anns[0]
    for ann in anns[1:]:
        merged_anns = merged_anns.merge(ann, conflicts=conflicts)
    return merged_anns


if __name__ == "__main__":
    args = parse_args()
    main(args)
