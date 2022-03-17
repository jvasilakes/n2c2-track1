import os
import json
import argparse
import datetime
from glob import glob
from collections import defaultdict

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config import ExperimentConfig
from data import DATAMODULE_LOOKUP
from data.n2c2 import n2c2ContextDataModule
from data.combined import CombinedDataModule
from models import MODEL_LOOKUP
from models.rationale import format_input_ids_and_masks

import brat_reader as br


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Suppress the tqdm progress bar")

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument(
            "config_file", type=str, help="Path to a yaml config file")

    val_parser = subparsers.add_parser(
            "validate", help="Evaluate a trained model on the dev set.")
    val_parser.add_argument(
            "config_file", type=str, help="Path to a yaml config file")
    val_parser.add_argument(
            "--dataset", type=str, default="dev", choices=["train", "dev"],
            help="Evaluate on this dataset. Default 'dev'.")
    val_parser.add_argument(
            "--output_brat", action="store_true", default=False,
            help="""If set, save brat formatted predictions
                    to the predictions/ directory under the logdir.""")
    val_parser.add_argument(
            "--output_token_masks", action="store_true", default=False,
            help="""If set, save tokens and stochastic masks to the
                    token_masks/ directory under the logdir.""")

    test_parser = subparsers.add_parser(
            "test", help="Evaluate a trained model on the test set.")
    test_parser.add_argument(
            "config_file", type=str, help="Path to a yaml config file")
    test_parser.add_argument(
            "--output_brat", action="store_true", default=False,
            help="""If set, save brat formatted predictions
                    to the predictions/ directory under the logdir.""")

    return parser.parse_args()


def main(args):
    config = ExperimentConfig.from_yaml_file(args.config_file)
    logdir = config.logdir
    os.makedirs(logdir, exist_ok=True)

    curr_time = datetime.datetime.now()
    print(f"Start: {curr_time}")
    print(config)

    pl.seed_everything(config.random_seed, workers=True)

    datamodule = load_datamodule_from_config(config)
    datamodule.setup()

    print("Label Spec")
    print('  ' + str(datamodule.label_spec))
    run_args = [config, datamodule]
    run_kwargs = {
            "logdir": logdir,
            "quiet": args.quiet,
            }
    if args.command == "train":
        version = get_next_experiment_version(logdir, config.name)
        run_kwargs["version"] = version
        run_fn = run_train
    elif args.command in ["validate", "test"]:
        run_kwargs["dataset"] = args.dataset
        version = get_current_experiment_version(args.config_file)
        run_kwargs["version"] = version
        run_kwargs["output_brat"] = args.output_brat
        run_kwargs["output_token_masks"] = args.output_token_masks
        run_fn = run_validate
    else:
        raise argparse.ArgumentError(f"Unknown command {args.command}")
    run_fn(*run_args, **run_kwargs)

    curr_time = datetime.datetime.now()
    print(f"End: {curr_time}")


def load_datamodule_from_config(config: ExperimentConfig):
    datamodule = n2c2ContextDataModule.from_config(config)
    if len(config.auxiliary_data) > 0:
        all_datamods = [datamodule]
        dm_names = [datamodule.name]
        for (dataset_name, kwargs) in config.auxiliary_data.items():
            datamodule_cls = DATAMODULE_LOOKUP[dataset_name]
            dm = datamodule_cls.from_config(config, **kwargs)
            if dm.name in dm_names:
                raise ValueError(f"Already loaded a datamodule '{dm.name}'! Make sure you're not using duplicate datasets or specify a unique name for any entries under `auxiliary_data`.")  # noqa
            all_datamods.append(dm)
            dm_names.append(dm.name)
        datamodule = CombinedDataModule(
            all_datamods, dataset_sample_strategy=config.dataset_sample_strategy)  # noqa
    return datamodule


def get_next_experiment_version(logdir, experiment_name):
    all_version_dirs = glob(os.path.join(logdir, experiment_name, "version_*"))
    if all_version_dirs == []:
        return 1
    all_version_dirs = [os.path.basename(d) for d in all_version_dirs]
    max_version = max([int(d.split('_')[-1]) for d in all_version_dirs])
    return max_version + 1


def get_current_experiment_version(config_path):
    # config.yaml is always saved directly under the version directory.
    # E.g., "logs/test/version_1/config.yaml"
    if "version_" not in config_path:
        err_str = f"""No version directory found within {config_path}.
         If you're running validation, pass the path to the logged
         config file. E.g., logs/experiment/version_1/config.yaml"""
        raise OSError(err_str)
    version_dirname = config_path.split('/')[-2]
    version_num = int(version_dirname.split('_')[-1])
    return version_num


def run_train(config, datamodule,
              logdir="logs/", version=None, quiet=False):

    version_dir = os.path.join(logdir, config.name, f"version_{version}")
    os.makedirs(version_dir, exist_ok=False)
    config.save_to_yaml(os.path.join(version_dir, "config.yaml"))

    model_class = MODEL_LOOKUP[config.model_name]
    model = model_class.from_config(config, datamodule)

    logger = TensorBoardLogger(
            save_dir=logdir, version=version, name=config.name)

    checkpoint_cb = ModelCheckpoint(
            monitor="avg_macro_f1",
            mode="max",
            filename="{epoch:02d}-{avg_macro_f1:.2f}")

    available_gpus = min(1, torch.cuda.device_count())
    enable_progress_bar = not quiet
    # There is a bug with deterministic indexing on the gpu
    #  in the current pytorch version, so we have to turn it off.
    #  https://github.com/pytorch/pytorch/issues/61032
    trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.max_epochs,
            gpus=available_gpus,
            gradient_clip_val=config.gradient_clip_val,
            deterministic=False,
            callbacks=[checkpoint_cb],
            enable_progress_bar=enable_progress_bar,
            log_every_n_steps=32,
            )
    trainer.fit(model, datamodule=datamodule)


def run_validate(config, datamodule, dataset="dev",
                 logdir="logs/", version=None, quiet=False,
                 output_brat=False, output_token_masks=False):
    # Find the checkpoint and hparams files.
    # Since we only checkpoint the best dev performance during training,
    #   there is only ever one checkpoint.
    checkpoint_dir = os.path.join(
            logdir, config.name, f"version_{version}", "checkpoints")
    checkpoint_file = glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if checkpoint_file == []:
        raise OSError(f"No checkpoints found in {checkpoint_dir}")
    hparams_file = os.path.join(checkpoint_dir, "../hparams.yaml")

    model_class = MODEL_LOOKUP[config.model_name]
    model = model_class.load_from_checkpoint(
            checkpoint_path=checkpoint_file[0],
            hparams_file=hparams_file)
    model.eval()

    available_gpus = min(1, torch.cuda.device_count())
    enable_progress_bar = not quiet
    trainer = pl.Trainer(
            logger=False,  # Disable tensorboard logging
            gpus=available_gpus,
            enable_progress_bar=enable_progress_bar
            )
    if dataset == "train":
        val_dataloader = datamodule.train_dataloader()
    elif dataset == "dev":
        val_dataloader = datamodule.val_dataloader()
    elif dataset == "test":
        if datamodule.test_dataloader() is None:
            raise OSError("No test data found. Aborting.")
        val_dataloader = datamodule.test_dataloader()
    else:
        raise ValueError(f"Unknown validation dataset '{dataset}'")
    results = trainer.validate(
        model, dataloaders=val_dataloader, verbose=False)[0]
    tasks = sorted(datamodule.label_spec.keys())
    md = format_results_as_markdown_table(results, tasks)
    print(md)

    if output_brat is True or output_token_masks is True:
        preds = trainer.predict(model, dataloaders=val_dataloader)
    if output_brat is True:
        train_dataset = datamodule.train
        # List[BratAnnotations]
        pred_anns_by_docid = batched_predictions_to_brat(preds, train_dataset)
        preds_dir = os.path.join(checkpoint_dir, "../predictions", dataset)
        os.makedirs(preds_dir, exist_ok=False)
        for doc_preds in pred_anns_by_docid:
            doc_preds.save_brat(preds_dir)

    if output_token_masks is True:
        if config.model_name != "bert-rationale-classifier":
            raise ValueError("--output_token_masks only compatible with bert-rationale-classifier.")  # noqa
        mask_dir = os.path.join(checkpoint_dir, "../token_masks", dataset)
        os.makedirs(mask_dir, exist_ok=False)
        masked_by_task = batched_predictions_to_masked_tokens(
                preds, datamodule)
        for (task, masked_examples) in masked_by_task.items():
            outpath = os.path.join(mask_dir, f"{task}.jsonl")
            with open(outpath, 'w') as outF:
                for datum in masked_examples:
                    json.dump(datum, outF)
                    outF.write('\n')


def format_results_as_markdown_table(results, tasks):
    """
    |task1 | P  | R | F1 | task2 | P | R | F1 |
    |------|----|---|----|-------|---|---|----|
    |micro |    |   |    | micro |   |   |    |
    |macro |    |   |    | macro |   |   |    |
    """
    task_abbrevs = {
            "Action": "Action",
            "Actor": "Actor",
            "Certainty": "Cert",
            "Negation": "Neg",
            "Temporality": "Temp"
            }
    # Header
    table = '|'
    for task in tasks:
        task_str = task_abbrevs[task]
        table += f" {task_str: <7} | {'P': <5} | {'R': <5} | {'F1': <5} |"

    # Separator
    table += "\n|"
    for task in tasks:
        table += "---------|-------|-------|-------|"

    for avg_fn in ["micro", "macro"]:
        table += "\n|"
        for task in tasks:
            p = results[f"{avg_fn}_{task}_precision"]
            r = results[f"{avg_fn}_{task}_recall"]
            f = results[f"{avg_fn}_{task}_F1"]
            table += f" {avg_fn: <7} | {p:.3f} | {r:.3f} | {f:.3f} |"
    table += '\n'
    return table


def batched_predictions_to_masked_tokens(preds, datamodule):
    masked_by_task = defaultdict(list)
    for batch in preds:
        for (i, docid) in enumerate(batch["docids"]):
            for (task, zmasks) in batch["zmask"].items():
                datum = {}
                datum["docid"] = docid
                input_ids = batch["input_ids"][i]
                datum["tokens_with_masks"] = format_input_ids_and_masks(
                        input_ids, zmasks[i], datamodule.tokenizer)
                enc_pred = batch["predictions"][task][i].int().item()
                enc_lab = batch["labels"][task][i].int().item()
                datum["prediction"] = datamodule.train.inverse_transform(
                        task, [enc_pred])[0]
                datum["label"] = datamodule.train.inverse_transform(
                        task, [enc_lab])[0]
                masked_by_task[task].append(datum)
    return masked_by_task


def batched_predictions_to_brat(preds, dataset):
    """
    Create a BratAnnotations instance for each grouped of predictions,
    grouped by doc_id.
    """
    events_by_docid = defaultdict(list)
    for batch in preds:
        for (i, docid) in enumerate(batch["docids"]):
            attrs = {}
            for (task, task_preds) in batch["predictions"].items():
                enc_pred = task_preds[i].int().item()
                decoded_pred = dataset.inverse_transform(task, [enc_pred])[0]
                num_attrs = len([attr for e in events_by_docid[docid]
                                 for attr in e.attributes]) + len(attrs)
                aid = f"A{num_attrs}"
                attr = br.Attribute(id=aid, type=task, value=decoded_pred)
                attrs[task] = attr

            # Reconstruct original character offsets
            start, end = batch["entity_spans"][i]
            entity_text = batch["texts"][i][start:end]
            # There's one entity mention that spans a newline
            entity_text = entity_text.replace('\n', ' ')
            start += batch["char_offsets"][i]
            end += batch["char_offsets"][i]

            # Remove the '@' entity markers if used.
            if dataset.mark_entities is True:
                entity_text = entity_text.strip('@')
                end -= 2

            num_spans = len(set([(e.span.start_index, e.span.end_index)
                                 for e in events_by_docid[docid]]))
            sid = f"T{num_spans}"
            span = br.Span(id=sid, start_index=start, end_index=end,
                           text=entity_text)

            src_file_str = f"{docid}.ann"
            eid = f"E{len(events_by_docid[docid])}"
            event = br.Event(id=eid, type="Disposition", span=span,
                             attributes=attrs, _source_file=src_file_str)
            events_by_docid[docid].append(event)

    anns_by_docid = [br.BratAnnotations.from_events(events)
                     for events in events_by_docid.values()]
    return anns_by_docid


if __name__ == "__main__":
    args = parse_args()
    main(args)
