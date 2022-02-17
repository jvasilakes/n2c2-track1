import os
import argparse
from glob import glob
from collections import Counter, defaultdict

from brat_reader import BratAnnotations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="Path to directory containing {train,dev,test}")
    parser.add_argument("--logdir", type=str, default='.',
                        help="Where to save detailed statistics.")
    return parser.parse_args()


def main(args):
    entities_by_dataset = {}
    datasets = ["train", "dev", "test"]
    for dataset in datasets:
        data_path = os.path.join(args.data_dir, dataset)
        if not os.path.isdir(data_path):
            print(f"Dataset {dataset} not found at {data_path}. Skipping...")
            continue
        anns = get_all_annotations(data_path)
        entity_tokens, E_counts, A_counts = count_anns(anns)
        entities_by_dataset[dataset] = entity_tokens
        print(f"{dataset.upper()}")
        print_count_info(entity_tokens, E_counts, A_counts)
        print()

    print("==================")
    print("Dataset comparison")
    print("==================")
    train_ents = set(entities_by_dataset["train"])
    print(f"Total train entities: {len(train_ents)}")
    for (dataset, val_ents) in entities_by_dataset.items():
        if dataset == "train":
            continue
        val_ents = set(val_ents)
        print(f"  Total {dataset} entities: {len(val_ents)}")
        unique_val = val_ents.difference(train_ents)
        print(f"  {dataset} entities not in train: {len(unique_val)}")
        logfile = os.path.join(args.logdir, f"{dataset}_unseen.txt")
        with open(logfile, 'w') as outF:
            for entity in unique_val:
                outF.write(f"{entity}\n")
        print(f"  Unseen entities written to {logfile}")
        print()


def get_all_annotations(data_path):
    glob_path = os.path.join(data_path, "*.ann")
    events = [event for annfile in glob(glob_path)
              for event in BratAnnotations.from_file(annfile).events]
    all_anns = BratAnnotations.from_events(events)
    return all_anns


def count_anns(anns):
    tokens = []
    E_counts = defaultdict(int)  # Count event types
    A_counts = defaultdict(lambda: defaultdict(int))

    for event in anns.events:
        norm_text = event.span.text.lower()
        tokens.append(norm_text)
        E_counts[event.type] += 1
        for attr in event.attributes.values():
            A_counts[attr.type][attr.value] += 1
    return tokens, E_counts, A_counts


def print_count_info(tokens, entity_counts, attr_counts):
    print("Events")
    for event_type, count in entity_counts.items():
        print(f"  {event_type}: {count}")
    for label, label_vals in attr_counts.items():
        print(label)
        for label_val, count in label_vals.items():
            print(f"  {label_val}: {count}")
    tok_counts = Counter(tokens)
    print("\nEntity mention data:")
    print(f" Number of unique entities (lowercased): {len(tok_counts)} / {len(tokens)} total mentions")  # noqa
    print(" 10 most common entities:")
    for (ent_text, count) in tok_counts.most_common(10):
        print(f"   {ent_text}: {count}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
