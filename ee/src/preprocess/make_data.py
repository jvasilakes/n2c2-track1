from preprocess_spacy_words import preprocess_spacy
from os.path import isfile, join
import json
import argparse


def main(args):
    # data_path = ["../../data/original_v3/train/","../../data/original_v3/dev/"]
    # data_path = ["../../data/split0/brat/train/", "../../data/brat/split0/dev/"]
    # spacy_path = "../../data/split0/spacy/"
    data_path = [args.datadir + "train/", args.datadir + "dev/"]

    stats = {}
    for i, path in enumerate(data_path):
        stats[path.split('/')[-2]] = preprocess_spacy(path, args.spacydir, 1)

    with open(join(args.spacydir, 'spacy_stats.txt'), "w") as st_out:
        st_out.write(json.dumps(stats))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #     parser.add_argument('--config', type=str)
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--spacydir', type=str)
    args = parser.parse_args()
    main(args)
