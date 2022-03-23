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
    parser.add_argument("--log_file", type=str, default=None,
                        help="Where to save detailed evaluation information.")
    return parser.parse_args()


def main(args):
    gold_glob_path = os.path.join(args.gold_dir, "*.ann")
    gold_files = glob(gold_glob_path)
    # Match gold to pred files
    pred_and_gold_files = []
    for gold_path in gold_files:
        bn = os.path.basename(gold_path)
        pred_path = os.path.join(args.predictions_dir, bn)
        if not os.path.isfile(pred_path):
            print(f"No predictions found for {bn}. Skipping...")
            continue
        pred_and_gold_files.append((pred_path, gold_path))

    all_preds_by_task = defaultdict(list)
    all_golds_by_task = defaultdict(list)
    for (pred_f, gold_f) in pred_and_gold_files:
        pred_anns = BratAnnotations.from_file(pred_f)
        gold_anns = BratAnnotations.from_file(gold_f)

        tasklist = sorted(set([a["_type"] for a in pred_anns._raw_attributes]))
        for task in tasklist:
            # We're only predicting over Disposition events
            preds = []
            for e in pred_anns.get_events_by_type("Disposition"):
                val = e.attributes[task].value
                preds.append(val)
            golds = []
            for e in gold_anns.get_events_by_type("Disposition"):
                val = e.attributes[task].value
                golds.append(val)
            all_preds_by_task[task].extend(preds)
            all_golds_by_task[task].extend(golds)

    for task in all_preds_by_task.keys():
        preds = all_preds_by_task[task]
        golds = all_golds_by_task[task]
        labels = sorted(set(golds))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
                    golds, preds, average="micro")
            macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
                    golds, preds, average="macro")
            lab_p, lab_r, lab_f, lab_s = precision_recall_fscore_support(
                    golds, preds, average=None, labels=labels)

        avg_table = format_avg_results(micro_p, micro_r, micro_f,
                                       macro_p, macro_r, macro_f)

        per_label_table = format_per_label_results(
                lab_p, lab_r, lab_f, lab_s, labels=labels)

        if args.log_file is None:
            print(f"### {task}")
            print(avg_table, end='')
            print(per_label_table)
        else:
            with open(args.log_file, 'a') as outF:
                outF.write(f"### {task}\n")
                outF.write(avg_table)
                outF.write(per_label_table)


def format_per_label_results(precs, recs, fs, supports, labels=[]):
    if labels == [] or len(labels) != len(precs):
        labels = [str(i) for i in range(len(precs))]

    max_chars = max([len(lab) for lab in labels]) + 1
    tab = f"|{' ': <{max_chars}}| {'prec': <4}  | {'rec': <3}   | {'f1': <4}  | {'supp': <4} |\n"  # noqa
    tab += f"|{'-' * max_chars}|-------|-------|-------|------|\n"

    for (l, p, r, f, s) in zip(labels, precs, recs, fs, supports):
        tab += f"|{l: <{max_chars}}| {p:.3f} | {r:.3f} | {f:.3f} | {s: <4} |\n"
    tab += '\n'
    return tab


def format_avg_results(micro_p, micro_r, micro_f,
                       macro_p, macro_r, macro_f):
    tab = f"|      | {'prec': <4}  | {'rec': <3}   | {'f1': <4}  |\n"
    tab += "|------|-------|-------|-------|\n"
    tab += f"|MICRO | {micro_p:.3f} | {micro_r:.3f} | {micro_f:.3f} |\n"
    tab += f"|MACRO | {macro_p:.3f} | {macro_r:.3f} | {macro_f:.3f} |\n\n"
    return tab


if __name__ == "__main__":
    args = parse_args()
    main(args)
