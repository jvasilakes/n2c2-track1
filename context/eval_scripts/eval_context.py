import os
import argparse
import warnings
from glob import glob
from collections import defaultdict

from sklearn.metrics import precision_recall_fscore_support

from brat_reader import BratAnnotations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_dir", type=str,
                        help="""Directory containing files with
                                predictions in brat format.""")
    parser.add_argument("gold_dir", type=str,
                        help="""Directory containing files with gold
                                labels in brat format.""")
    parser.add_argument("--log_file", type=str, default="eval.log",
                        help="Where to save detailed evaluation information.")
    return parser.parse_args()


def main(args):
    pred_glob_path = os.path.join(args.predictions_dir, "*.ann")
    pred_files = glob(pred_glob_path)
    gold_files = [os.path.join(args.gold_dir, os.path.basename(pred_file))
                  for pred_file in pred_files]

    all_preds_by_task = defaultdict(list)
    all_golds_by_task = defaultdict(list)
    for (pred_f, gold_f) in zip(pred_files, gold_files):
        pred_anns = BratAnnotations.from_file(pred_f)
        gold_anns = BratAnnotations.from_file(gold_f)

        tasklist = sorted(set([a["type"] for a in pred_anns._raw_attributes]))
        for task in tasklist:
            # We're only predicting over Disposition events
            preds = [e.attributes[task].value for e in pred_anns.events]
            golds = []
            for e in gold_anns.get_events_by_type("Disposition"):
                try:
                    val = e.attributes[task].value
                except KeyError:
                    if task == "Negation":
                        val = "NotNegated"
                golds.append(val)
            all_preds_by_task[task].extend(preds)
            all_golds_by_task[task].extend(golds)

    for task in all_preds_by_task.keys():
        preds = all_preds_by_task[task]
        golds = all_golds_by_task[task]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
                    golds, preds, average="micro")
            macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
                    golds, preds, average="macro")
        print(f"### {task}                     ")
        print(f"|      | {'prec': <4}  | {'rec': <3}   | {'f1': <4}  |")
        print("|------|-------|-------|-------|")
        print(f"|MICRO | {micro_p:.3f} | {micro_r:.3f} | {micro_f:.3f} |")
        print(f"|MACRO | {macro_p:.3f} | {macro_r:.3f} | {macro_f:.3f} |")
        print()


if __name__ == "__main__":
    args = parse_args()
    main(args)
