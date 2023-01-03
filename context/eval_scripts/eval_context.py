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
    parser.add_argument("--mode", type=str, default="lenient",
                        choices=["strict", "lenient"],
                        help="Span matching mode: strict or lenient.")
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
        pred_anns, gold_anns = align_annotations(pred_anns, gold_anns,
                                                 mode=args.mode)
        for task in tasklist:
            preds = []
            for e in pred_anns:
                try:
                    val = e.attributes[task].value
                except AttributeError:
                    val = "null"
                preds.append(val)
            golds = []
            for e in gold_anns:
                try:
                    val = e.attributes[task].value
                except AttributeError:
                    val = "null"
                golds.append(val)
            all_preds_by_task[task].extend(preds)
            all_golds_by_task[task].extend(golds)

    for task in all_preds_by_task.keys():
        preds = all_preds_by_task[task]
        golds = all_golds_by_task[task]
        labels = sorted(set([g for g in golds if g != "null"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
                    golds, preds, average="micro", labels=labels)
            macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
                    golds, preds, average="macro", labels=labels)
            lab_p, lab_r, lab_f, lab_s = precision_recall_fscore_support(
                    golds, preds, average=None, labels=labels)

        avg_table = format_avg_results(micro_p, micro_r, micro_f,
                                       macro_p, macro_r, macro_f)

        per_label_table = format_per_label_results(
                lab_p, lab_r, lab_f, lab_s, labels=labels)

        print(f"Mode: {args.mode}")
        if args.log_file is None:
            print(f"### {task}")
            print(avg_table, end='')
            print(per_label_table)
        else:
            with open(args.log_file, 'a') as outF:
                outF.write(f"### {task}\n")
                outF.write(avg_table)
                outF.write(per_label_table)


def align_annotations(preds, golds, mode="lenient"):
    # We're only predicting over Disposition events
    pred_disp = preds.get_events_by_type("Disposition")
    gold_disp = golds.get_events_by_type("Disposition")
    aligned_preds = []
    aligned_golds = []
    matched_golds = set()
    for p in pred_disp:
        matched = False
        for g in gold_disp:
            if indices_overlap(p, g, mode=mode) is True:
                aligned_preds.append(p)
                aligned_golds.append(g)
                matched = True
                matched_golds.add(g)
                break
        if matched is False:
            aligned_preds.append(p)
            aligned_golds.append(None)
    for g in gold_disp:
        if g not in matched_golds:
            aligned_preds.append(None)
            aligned_golds.append(g)
    return aligned_preds, aligned_golds


def indices_overlap(a, b, mode="lenient"):
    if mode == "lenient":
        idx_range = range(b.start_index, b.end_index)
        if a.start_index in idx_range or a.end_index in idx_range:
            return True
    elif mode == "strict":
        if (a.start_index, a.end_index) == (b.start_index, b.end_index):
            return True
    return False


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
