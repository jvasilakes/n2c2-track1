import os
import argparse
from glob import glob
from collections import defaultdict


"""
Compute micro and macro-averaged precision, recall, and F1 score between
predicted NER character spans and gold-standard spans. Computes
all metrics using both exact span matching and lenient matching (minimum
1 character overlap). Assumes brat formatted files.

Usage:
    python3 ner.py /path/to/predictions/dir /path/to/gold/dir

Filenames in the predictions dir should match the pattern
    ([0-9a-zA-Z\-_]+)\.?.*\.ann
    where everything captured by the first group is a unique identifier
    that can be used to find the corresponding gold file, per below.
    E.g., 100-01.txt.ann
Filenames in the gold dir should match
    [0-9a-zA-Z\-_]+\.ann
    E.g., 100-01.ann
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_dir", type=str,
                        help="""Directory containing files
                                with predictions in brat format.""")
    parser.add_argument("gold_dir", type=str,
                        help="""Directory containing files with gold labels
                                in brat format.""")
    return parser.parse_args()


def main(args):
    glob_path = os.path.join(args.predictions_dir, "*.ann")
    glob_files = glob(glob_path)
    if len(glob_files) == 0:
        raise OSError(f"No input files found at {glob_path}")

    totals = defaultdict(int)
    metrics = defaultdict(list)
    for pred_fname in glob_files:
        pred_bn = os.path.basename(pred_fname)
        uid = pred_bn.split('.', maxsplit=1)[0]
        gold_fname = f"{uid}.ann"
        gold_path = os.path.join(args.gold_dir, gold_fname)

        pred_spans = get_spans_from_anns(pred_fname)
        gold_spans = get_spans_from_anns(gold_path)

        tp, fp, fn = evaluate_ner(pred_spans, gold_spans, lenient=False)
        totals["exact_tp"] += tp
        totals["exact_fp"] += fp
        totals["exact_fn"] += fn
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        metrics["exact_precision"].append(p)
        metrics["exact_recall"].append(r)
        metrics["exact_f1"].append(f1)

        tp, fp, fn = evaluate_ner(pred_spans, gold_spans, lenient=True)
        totals["lenient_tp"] += tp
        totals["lenient_fp"] += fp
        totals["lenient_fn"] += fn
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        metrics["lenient_precision"].append(p)
        metrics["lenient_recall"].append(r)
        metrics["lenient_f1"].append(f1)

    micro_p, micro_r, micro_f1 = precision_recall_f1(
            totals["exact_tp"], totals["exact_fp"], totals["exact_fn"])
    denom = len(metrics["exact_precision"])
    macro_p = sum(metrics["exact_precision"]) / denom
    macro_r = sum(metrics["exact_recall"]) / denom
    macro_f1 = sum(metrics["exact_f1"]) / denom

    print("### Exact match")
    print(f"|      | {'prec': <4}  | {'rec': <3}   | {'f1': <4}  |")
    print("|------|-------|-------|-------|")
    print(f"|MICRO | {micro_p:.3f} | {micro_r:.3f} | {micro_f1:.3f} |")
    print(f"|MACRO | {macro_p:.3f} | {macro_r:.3f} | {macro_f1:.3f} |")

    micro_p, micro_r, micro_f1 = precision_recall_f1(
            totals["lenient_tp"], totals["lenient_fp"], totals["lenient_fn"])
    macro_p = sum(metrics["lenient_precision"]) / denom
    macro_r = sum(metrics["lenient_recall"]) / denom
    macro_f1 = sum(metrics["lenient_f1"]) / denom

    print()
    print("### Lenient match")
    print(f"|      | {'prec': <4}  | {'rec': <3}   | {'f1': <4}  |")
    print("|------|-------|-------|-------|")
    print(f"|MICRO | {micro_p:.3f} | {micro_r:.3f} | {micro_f1:.3f} |")
    print(f"|MACRO | {macro_p:.3f} | {macro_r:.3f} | {macro_f1:.3f} |")


def get_spans_from_anns(ann_file):
    spans = set()
    with open(ann_file, 'r') as inF:
        for line in inF:
            line = line.strip()
            if line[0] != 'T':
                continue
            uid, _, rest = line.split(maxsplit=2)
            if ';' not in rest:
                start_i, end_i, text = rest.split(maxsplit=2)
            else:
                continue
            spans.add((int(start_i), int(end_i), text))
    return spans


def evaluate_ner(preds, golds, lenient=False):
    if lenient is False:
        tp = len(preds.intersection(golds))
        fp = len(preds.difference(golds))
        fn = len(golds.difference(preds))
    else:
        tp = lenient_intersection(preds, golds)
        fp = lenient_difference(preds, golds)
        fn = lenient_difference(golds, preds)
    return tp, fp, fn


def precision_recall_f1(tp, fp, fn):
    if (tp + fp) == 0:
        p = 0.
    else:
        p = tp / (tp + fp)

    if (tp + fn) == 0:
        r = 0.
    else:
        r = tp / (tp + fn)

    if (p + r) == 0:
        f1 = 0.
    else:
        f1 = 2 * (p * r) / (p + r)

    return p, r, f1


def lenient_intersection(spans1, spans2):
    """
    Compute the size of the set intersection between
    spans1 and spans2 where spans match if
    1 or more characters overlap.
    """
    total = 0
    for s1 in spans1:
        for s2 in spans2:
            if overlap(s1, s2, min_chars=1) is True:
                total += 1
    return total


def lenient_difference(spans1, spans2):
    """
    Compute the size of the set difference between
    spans1 and spans2 where spans match if
    1 or more characters overlap.
    """
    total = 0
    for s1 in spans1:
        match = False
        for s2 in spans2:
            if overlap(s1, s2, min_chars=1) is True:
                match = True
                break
        if match is False:
            total += 1
    return total


def overlap(span1, span2, min_chars=1):
    """
    Determine if two spans overlap by at least min_chars.
    """
    s1_start, s1_end, _ = span1
    s2_start, s2_end, _ = span2
    if s1_start <= s2_start:
        if s1_end - s2_start >= min_chars:
            return True
    if s2_start <= s1_start:
        if s2_end - s1_start >= min_chars:
            return True
    return False


def test_lenient_intersection():
    spans1 = [(1, 5, ""), (9, 12, ""), (13, 14, "")]
    spans2 = [(9, 12, ""), (13, 15, ""), (19, 22, "")]
    intersect = lenient_intersection(spans1, spans2)
    assert intersect == 2
    intersect = lenient_intersection(spans2, spans1)
    assert intersect == 2

    spans1 = [(1, 5, "")]
    spans2 = [(9, 12, ""), (13, 15, ""), (19, 22, "")]
    intersect = lenient_intersection(spans1, spans2)
    assert intersect == 0
    intersect = lenient_intersection(spans2, spans1)
    assert intersect == 0

    spans1 = [(9, 12, ""), (13, 15, ""), (19, 22, "")]
    spans2 = [(9, 12, ""), (13, 15, ""), (19, 22, "")]
    intersect = lenient_intersection(spans1, spans2)
    assert intersect == 3
    intersect = lenient_intersection(spans2, spans1)
    assert intersect == 3

    spans1 = []
    spans2 = []
    intersect = lenient_intersection(spans1, spans2)
    assert intersect == 0
    intersect = lenient_intersection(spans2, spans1)
    assert intersect == 0

    print("test_lenient_intersection passed")


def test_lenient_difference():
    spans1 = [(1, 5, ""), (9, 12, ""), (13, 14, "")]
    spans2 = [(9, 12, ""), (13, 15, ""), (19, 22, "")]
    diff = lenient_difference(spans1, spans2)
    assert diff == 1
    diff = lenient_difference(spans2, spans1)
    assert diff == 1

    spans1 = [(1, 5, "")]
    spans2 = [(9, 12, ""), (13, 15, ""), (19, 22, "")]
    diff = lenient_difference(spans1, spans2)
    assert diff == 1
    diff = lenient_difference(spans2, spans1)
    assert diff == 3

    spans1 = [(9, 12, ""), (13, 15, ""), (19, 22, "")]
    spans2 = [(9, 12, ""), (13, 15, ""), (19, 22, "")]
    diff = lenient_difference(spans1, spans2)
    assert diff == 0
    diff = lenient_difference(spans2, spans1)
    assert diff == 0

    spans1 = []
    spans2 = []
    diff = lenient_difference(spans1, spans2)
    assert diff == 0
    diff = lenient_difference(spans2, spans1)
    assert diff == 0

    print("test_lenient_difference passed")


def test_overlap():
    span1 = (1, 5, "")
    span2 = (4, 8, "")
    assert overlap(span1, span2) is True

    span1 = (4, 8, "")
    span2 = (1, 5, "")
    assert overlap(span1, span2) is True

    span1 = (1, 8, "")
    span2 = (1, 5, "")
    assert overlap(span1, span2) is True

    span1 = (9, 12, "")
    span2 = (9, 12, "")
    assert overlap(span1, span2) is True

    span1 = (1, 3, "")
    span2 = (4, 8, "")
    assert overlap(span1, span2) is False

    span1 = (4, 8, "")
    span2 = (1, 3, "")
    assert overlap(span1, span2) is False

    print("test_overlap passed")


if __name__ == "__main__":
    args = parse_args()
    main(args)
