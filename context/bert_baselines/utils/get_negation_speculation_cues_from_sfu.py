import os
import argparse
from glob import glob
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, required=True,
                        help="path to SFU_Review_Corpus_Negation_Speculation")
    parser.add_argument("--outdir", type=str, required=True,
                        help="where to save the word lists.")
    return parser.parse_args()


def main(args):
    topics = ['MOVIES', 'HOTELS', 'PHONES', 'BOOKS',
              'COOKWARE', 'MUSIC', 'CARS', 'COMPUTERS']
    os.makedirs(args.outdir, exist_ok=False)

    all_neg_cues = set()
    all_spec_cues = set()
    for topic in topics:
        xml_glob = os.path.join(args.indir, topic, "*.xml")
        for fname in glob(xml_glob):
            negs, specs = get_negation_speculation_cues(fname)
            all_neg_cues.update(negs)
            all_spec_cues.update(specs)

    neg_outfile = os.path.join(args.outdir, "negation_cues.txt")
    with open(neg_outfile, 'w') as outF:
        for cue in all_neg_cues:
            outF.write(f"{cue}\n")

    spec_outfile = os.path.join(args.outdir, "speculation_cues.txt")
    with open(spec_outfile, 'w') as outF:
        for cue in all_spec_cues:
            outF.write(f"{cue}\n")


def get_negation_speculation_cues(xmlfile):
    negs_out = set()
    specs_out = set()

    tree = ET.parse(xmlfile)
    root = tree.getroot()
    cues = root.findall(".//cue")
    for cue in cues:
        cue_type = cue.attrib["type"]
        w = cue.find("./W")
        if w is not None:
            text = w.text.lower()
            if cue_type == "negation":
                negs_out.add(text)
            elif cue_type == "speculation":
                specs_out.add(text)
            else:
                raise KeyError(f"Unsupported cue type {cue_type}")
    return negs_out, specs_out


if __name__ == "__main__":
    args = parse_args()
    main(args)
