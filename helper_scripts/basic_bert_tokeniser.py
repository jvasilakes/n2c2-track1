import json
import os
import argparse

from transformers import BasicTokenizer



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default='Jake_github/n2c2-track1/n2c2Track1TrainingData/segmented/train',
                        help="Directory containing .json files. (output by biomedicus)")
    parser.add_argument("--outdir", type=str, default='Jake_github/n2c2-track1/n2c2Track1TrainingData/tokenised/train',
                        help="Where to save tokenized files.")
    
    return parser.parse_args()


def tokenize_file(fileIn, fileOut, tokenizer):
    data = json.load(open(fileIn))
    sentences = data["documents"]["plaintext"]["label_indices"]["sentences"]["labels"]
    for sentence in sentences:            
        start_sent = sentence['start_index']
        # end_sent = sentence['end_index']
        tokens = tokenizer.tokenize(sentence['_text'])          
        result = []
        start_find = 0
        for token in tokens:
            s_tok = start_sent + sentence['_text'].index(token, start_find)
            result.append((s_tok, s_tok  +  len(token), token)) 
            start_find += len(token)
        
        sentence['tokens'] = result
    
    with open(fileOut, "w", encoding="utf8") as fw_p:
        fw_p.write(json.dumps(data))

def main(args):
    tokenizer = BasicTokenizer(do_lower_case=False)
    for file in os.listdir(args.indir):
        if not 'json' in file:
            continue
        fileIn = os.path.join(args.indir, file)
        fileOut = os.path.join(args.outdir, file)
        tokenize_file (fileIn, fileOut, tokenizer)


if __name__ == "__main__":
    args = parse_args()
    main(args)