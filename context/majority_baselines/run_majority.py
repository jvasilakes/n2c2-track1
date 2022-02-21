import os
import argparse
import warnings
from glob import glob
from collections import defaultdict

from brat_reader import BratAnnotations

"""
Always predict the majority label for each context classification task.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str,
                        help="Directory containing {train/dev}*.ann files.")
    parser.add_argument("outdir", type=str,
                        help="Where to save the predictions.")
    return parser.parse_args()


def main(args):
    train_path = os.path.join(args.datadir, "train")
    if not os.path.isdir(train_path):
        msg = f"Train dataset not found at {train_path}. Aborting!"
        raise OSError(msg)

    annglob = os.path.join(train_path, "*.ann")
    all_events = []
    for annfile in glob(annglob):
        anns = BratAnnotations.from_file(annfile)
        all_events.extend(anns.events)

    # {task: label} where label is the most common label for task.
    majority_labels = get_most_common_labels(all_events)

    for dataset in ["train", "dev", "test"]:
        dataset_path = os.path.join(args.datadir, dataset)
        if not os.path.isdir(dataset_path):
            msg = f"{dataset} datset not found at {dataset_path}. Skipping..."
            warnings.warn(msg)
            continue

        dataset_outdir = os.path.join(args.outdir, dataset)
        os.makedirs(dataset_outdir, exist_ok=False)
        annglob = os.path.join(dataset_path, "*.ann")
        for annfile in glob(annglob):
            anns = BratAnnotations.from_file(annfile)
            for event in anns.get_events_by_type("Disposition"):
                for (task, maj_label) in majority_labels.items():
                    event.attributes[task].update("value", maj_label)
            anns.save_brat(dataset_outdir)


def get_most_common_labels(events):
    label_counts = defaultdict(lambda: defaultdict(int))
    for event in events:
        for (task, attr) in event.attributes.items():
            label_counts[task][attr.value] += 1

    majority_labels = {}
    for (task, counts) in label_counts.items():
        majority_labels[task] = max(counts.items(), key=lambda x: x[1])[0]
    return majority_labels


if __name__ == "__main__":
    args = parse_args()
    main(args)
