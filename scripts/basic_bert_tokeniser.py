import json
import os
import argparse
from tqdm import tqdm

from transformers import BasicTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str,
                        default='n2c2Track1TrainingData/segmented/train',  # noqa
                        help="Directory containing sentence .json files.")
    parser.add_argument("--outdir", type=str,
                        default='n2c2Track1TrainingData/tokenised/train',  # noqa
                        help="Where to save tokenized files.")
    return parser.parse_args()


def tokenize_file(fileIn, fileOut, tokenizer):
    with open(fileIn, 'r') as inF:
        sentences = [json.loads(line.strip()) for line in inF]
    for sentence in sentences:
        start_sent = sentence['start_index']
        # end_sent = sentence['end_index']
        tokens = tokenizer.tokenize(sentence['_text'])
        result = []
        start_find = 0
        for token in tokens:
            while sentence["_text"][start_find] == ' ':
                start_find += 1
            s_tok = start_sent + sentence['_text'].index(token, start_find)
            result.append((s_tok, s_tok + len(token), token))
            start_find += len(token)

        sentence['tokens'] = result
        with open(fileOut, 'a', encoding='utf8') as fw_p:
            json.dump(sentence, fw_p)
            fw_p.write('\n')


def main(args):
    os.makedirs(args.outdir, exist_ok=False)
    tokenizer = BasicTokenizer(do_lower_case=False)
    for fpath in tqdm(os.listdir(args.indir)):
        if 'json' not in fpath:
            continue
        fileIn = os.path.join(args.indir, fpath)
        fileOut = os.path.join(args.outdir, fpath)
        tokenize_file(fileIn, fileOut, tokenizer)


if __name__ == "__main__":
    args = parse_args()
    main(args)
