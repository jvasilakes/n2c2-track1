import argparse

from src.ensembling import ENSEMBLE_LOOKUP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dirs", nargs='+', required=True,
                        help="Paths to models to ensemble.")
    parser.add_argument("--dataset", type=str, default="n2c2ContextDataset")
    parser.add_argument("--datasplit", default="dev",
                        choices=["train", "dev", "test"])
    parser.add_argument("--task", type=str, default="Action")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--ensemble_method", default="max-voting",
                        choices=["max-voting"])
    return parser.parse_args()


def main(args):
    ensembler_cls = ENSEMBLE_LOOKUP[args.ensemble_method]
    ensembler = ensembler_cls(
            args.model_dirs, args.dataset, args.datasplit, args.task)
    predictions = ensembler.run()
    for (fname, anns) in predictions.items():
        anns.save_brat(args.outdir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
