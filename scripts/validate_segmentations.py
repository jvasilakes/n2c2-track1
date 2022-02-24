import os
import json
import argparse

from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments_dir", type=str, required=True,
                        help="""JSON lines file containing segmentations
                                for a single document.""")
    parser.add_argument("--text_dir", type=str, required=True,
                        help="Text file to validate segmentations against.")
    return parser.parse_args()


def main(args):
    segments_files = glob(os.path.join(args.segments_dir, "*.json"))
    os.makedirs("omissions", exist_ok=True)

    num_omitted = 0
    for sfile in segments_files:
        with open(sfile, 'r') as inF:
            segments = [json.loads(line.strip()) for line in inF]
        txt_fname = os.path.basename(sfile).replace(".json", '')
        txt_path = os.path.join(args.text_dir, txt_fname)
        with open(txt_path, 'r') as inF:
            text = inF.read()

        reconstructed, omitted = concatenate_and_fill(segments, text)
        if len(omitted) > 0:
            num_omitted += 1
        omissions_file = os.path.join("omissions", os.path.basename(sfile))
        with open(omissions_file, 'w') as outF:
            for omission in omitted:
                outF.write(omission + '\n')

    print(f"Number of files with omissions: {num_omitted}")
    print("Any omissions written to omissions/")


def concatenate_and_fill(segments, text):
    reconstructed = ''
    curr_char_index = 0
    omitted = []
    for seg in segments:
        if seg["start_index"] > curr_char_index:
            to_add = text[curr_char_index:seg["start_index"]]
            if to_add.strip() != '':
                omitted.append(to_add)
            reconstructed += to_add
        reconstructed += seg["_text"]
        curr_char_index = seg["end_index"]
    if len(text) > curr_char_index:
        to_add = text[curr_char_index:len(text)]
        if to_add.strip() != '':
            omitted.append(to_add)
        reconstructed += to_add
    return reconstructed, omitted


if __name__ == "__main__":
    args = parse_args()
    main(args)
