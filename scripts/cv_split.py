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
    subparsers = parser.add_subparsers(dest="command")

    new_parser = subparsers.add_parser(
            "new", help="Create a new set of cv splits")
    new_parser.add_argument(
            "--indir", type=str, required=True,
            help="Directory containing train and dev splits.")
    new_parser.add_argument(
            "--outdir", type=str, required=True,
            help="Where to save the new train/dev splits.")
    new_parser.add_argument(
            "--num_splits", type=int, required=True,
            help="Number of different splits to generate.")
    new_parser.add_argument("--random_state", type=int, default=0)
    new_parser.add_argument(
            "--num_train", type=int, default=350,
            help="""Number of train documents to include in
                    the new split. Must be <= 400.""")

    apply_parser = subparsers.add_parser(
            "apply",
            help="""Apply an existing set of splits to a directory of files.
                    The files must have a common basename (minus the extension)
                    with the files in the existing splits.""")
    apply_parser.add_argument(
            "--split_dir", type=str, required=True,
            help="""Directory containing N train/dev splits
                    as output by cv_split.py new""")
    apply_parser.add_argument(
            "--indir", type=str, required=True,
            help="""Directory containing train/ and dev/
                    subdirectories to apply the splits to.""")
    apply_parser.add_argument(
            "--infile_ext", type=str, required=True,
            help="""The file extension for the files in indir.
                    E.g., --infile_ext '.json'""")
    return parser.parse_args()


def main(args):
    if args.command == "new":
        make_new_splits(args)
    elif args.command == "apply":
        apply_splits(args)
    else:
        raise ValueError(f"Unknown command '{args.command}'")


def make_new_splits(args):
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
        train = train_ann + train_txt
        dev = dev_ann + dev_txt
        save_split(train, dev, outdir)
    print(f"New splits saved to {args.outdir}")


def apply_splits(args):
    train_dir = os.path.join(args.indir, "train")
    dev_dir = os.path.join(args.indir, "dev")
    trg_trainfiles = get_files(train_dir, ext=args.infile_ext)
    trg_devfiles = get_files(dev_dir, ext=args.infile_ext)
    trg_basenames_to_paths = {
            os.path.basename(f).replace(args.infile_ext, ''): f
            for f in trg_trainfiles + trg_devfiles
            }

    splits = os.listdir(args.split_dir)
    for split in splits:
        # Skip any non-split directories/files.
        try:
            int(split)
        except ValueError:
            continue
        traindir = os.path.join(args.split_dir, split, "train")
        devdir = os.path.join(args.split_dir, split, "dev")
        src_trainfiles = get_files(traindir, ext=".ann")
        src_devfiles = get_files(devdir, ext=".ann")
        src_train_bns = [os.path.basename(f).replace(".ann", '')
                         for f in src_trainfiles]
        src_dev_bns = [os.path.basename(f).replace(".ann", '')
                       for f in src_devfiles]

        train = [trg_basenames_to_paths[bn] for bn in src_train_bns]
        dev = [trg_basenames_to_paths[bn] for bn in src_dev_bns]
        basedir = os.path.basename(args.indir.rstrip('/'))
        outdir = os.path.join(args.split_dir, split, basedir)
        save_split(train, dev, outdir)


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

    for fpath in trainpaths:
        shutil.copy2(fpath, train_dir)
    for fpath in devpaths:
        shutil.copy2(fpath, dev_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
