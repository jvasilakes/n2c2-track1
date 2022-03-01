import os
import shutil
import argparse
from glob import glob

from sklearn.model_selection import train_test_split


"""
Generate new train/dev splits given the original n2c2 data.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str,
                        help="Directory containing train and dev splits.")
    parser.add_argument("--outdir", type=str,
                        help="Where to save the new train/dev splits.")
    parser.add_argument("--num_splits", type=int,
                        help="Number of different splits to generate.")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--num_train", type=int, default=350,
                        help="""Number of train documents to include in
                                the new split. Must be <= 400.""")
    return parser.parse_args()


def main(args):
    train_dir = os.path.join(args.indir, "train")
    train_annfiles = get_files(train_dir, ext=".ann")
    train_txtfiles = get_files(train_dir, ext=".txt")
    dev_dir = os.path.join(args.indir, "dev")
    dev_annfiles = get_files(dev_dir, ext=".ann")
    dev_txtfiles = get_files(dev_dir, ext=".txt")

    all_annfiles = train_annfiles + dev_annfiles
    all_txtfiles = train_txtfiles + dev_txtfiles
    if len(all_annfiles) != len(all_txtfiles):
        raise OSError("Found different number of ann and txt files!")

    for (annfile, txtfile) in zip(all_annfiles, all_txtfiles):
        ann_id = os.path.basename(annfile).replace(".ann", '')
        txt_id = os.path.basename(txtfile).replace(".txt", '')
        if ann_id != txt_id:
            print(ann_id)
            print(txt_id)
            raise ValueError("ann files and txt files not in the same order!")

    for i in range(args.num_splits):
        train_ann, dev_ann, train_txt, dev_txt = train_test_split(
                all_annfiles, all_txtfiles,
                train_size=args.num_train, shuffle=True,
                random_state=i)
        outdir = os.path.join(args.outdir, str(i))
        train = zip(train_ann, train_txt)
        dev = zip(dev_ann, dev_txt)
        save_split(train, dev, outdir)
    print(f"New splits saved to {args.outdir}")


def get_files(path, ext=''):
    ext = '*' + ext
    files = glob(os.path.join(path, ext))
    files = sorted(files, key=lambda x: os.path.basename(x))
    if len(files) == 0:
        raise OSError(f"No files with extension '{ext}' found at {path}")
    return files


def save_split(trainpaths, devpaths, outdir):
    os.makedirs(outdir, exist_ok=False)
    train_dir = os.path.join(outdir, "train")
    dev_dir = os.path.join(outdir, "dev")
    os.makedirs(train_dir)
    os.makedirs(dev_dir)

    for (ann, txt) in trainpaths:
        shutil.copy2(ann, train_dir)
        shutil.copy2(txt, train_dir)
    for (ann, txt) in devpaths:
        shutil.copy2(ann, dev_dir)
        shutil.copy2(txt, dev_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
