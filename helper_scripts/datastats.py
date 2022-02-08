import os
import argparse
from glob import glob
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", nargs='+', type=str)
    return parser.parse_args()


def main(args):
    glob_paths = []
    for datadir in args.data_dirs:
        glob_path = os.path.join(datadir, "*.ann")
        glob_paths.extend(glob(glob_path))
    all_anns_by_type = defaultdict(list)
    for annfile in glob_paths:
        ann_data = process_ann_file(annfile)
        for ann_type, anns in ann_data.items():
            all_anns_by_type[ann_type].extend(anns)
    tokens, E_counts, A_counts = count_anns(all_anns_by_type)
    for event_type, count in E_counts.items():
        print(f"{event_type}: {count}")
    for label, label_vals in A_counts.items():
        print(label)
        for label_val, count in label_vals.items():
            print(f"  {label_val}: {count}")
    tok_counts = Counter(tokens)
    print("\nEntity mention data:")
    print(f" Number of unique entities (lowercased): {len(tok_counts)} / {len(tokens)} total mentions")  # noqa
    print(" 10 most common entities:")
    for (ent_text, count) in tok_counts.most_common(10):
        print(f"   {ent_text}: {count}")


def process_ann_file(annfile):
    anns_by_type = defaultdict(list)
    for line in open(annfile, 'r'):
        line = line.strip()
        ann_type = line[0]  # T, E, or A
        if ann_type == 'T':
            data = process_text(line)
        elif ann_type == 'E':
            data = process_event(line)
        elif ann_type == 'A':
            data = process_attribute(line)
        else:
            raise ValueError(f"Unsupported type '{ann_type}' in {annfile}.")  # noqa
        anns_by_type[ann_type].append(data)
    return anns_by_type


def process_text(line):
    uid, label, other = line.split(maxsplit=2)
    if ';' not in other:
        start_idx, end_idx, text = other.split(maxsplit=2)
        idxs = [(start_idx, end_idx)]
    else:
        text = ''
        idxs = []
        spans = other.split(';')
        for span in spans:
            start_idx, end_idx_plus = span.split(maxsplit=1)
            end_idx_split = end_idx_plus.split(maxsplit=1)
            if len(end_idx_split) > 1:
                end_idx, text = end_idx_split
            else:
                end_idx = end_idx_split[0]
            idxs.append((start_idx, end_idx))

    return {"ID": uid,
            "label": label,
            "idxs": idxs,
            "text": text}


def process_event(line):
    fields = line.split()
    assert len(fields) == 2
    uid, label_and_ref = fields
    label, ref = label_and_ref.split(':')
    return {"ID": uid,
            "label": label,
            "ref_span_id": ref}


def process_attribute(line):
    fields = line.split()
    if fields[1] == "Negation":
        fields.append(True)
    assert len(fields) == 4
    uid, label, ref, value = fields
    return {"ID": uid,
            "label": label,
            "ref_event_id": ref,
            "label_value": value}


def count_anns(anns_by_type):
    tokens = []
    E_counts = defaultdict(int)  # Count event types
    A_counts = defaultdict(lambda: defaultdict(int))
    for ann_type, anns in anns_by_type.items():
        for ann in anns:
            if ann_type == 'T':
                norm_text = ann["text"].lower()
                tokens.append(norm_text)
            elif ann_type == 'E':
                E_counts[ann["label"]] += 1
            elif ann_type == 'A':
                label = ann["label"]
                value = ann["label_value"]
                A_counts[label][value] += 1
    return tokens, E_counts, A_counts


if __name__ == "__main__":
    args = parse_args()
    main(args)
