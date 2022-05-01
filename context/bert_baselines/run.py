import os
import re
import json
import argparse
import datetime
import warnings
from glob import glob
from collections import defaultdict

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import brat_reader as br

from src.config import ExperimentConfig
from src.data import load_datamodule_from_config
from src.data.combined import CombinedDataModule
from src.models import MODEL_LOOKUP
from src.models.rationale import format_input_ids_and_masks


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Suppress the tqdm progress bar")

    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument(
            "config_file", type=str, help="Path to a yaml config file")
    train_parser.add_argument(
            "--save_last_epoch", action="store_true", default=False,
            help="Save checkpoint after last training epoch.")

    cont_parser = subparsers.add_parser("continue", help="Continue training.")
    cont_parser.add_argument(
            "config_file", type=str, help="Path to a yaml config file")
    cont_parser.add_argument(
            "--save_last_epoch", action="store_true", default=False,
            help="Save checkpoint after last training epoch.")

    val_parser = subparsers.add_parser(
            "validate", help="Evaluate a trained model.")
    val_parser.add_argument(
            "config_file", type=str, help="Path to a yaml config file")
    val_parser.add_argument(
            "--datasplit", type=str, default="dev",
            choices=["train", "dev", "test"],
            help="Evaluate on this data split. Default 'dev'.")

    pred_parser = subparsers.add_parser(
            "predict", help="Run prediction on the specified data split.")
    pred_parser.add_argument(
            "config_file", type=str, help="Path to a yaml config file.")
    pred_parser.add_argument(
            "--datasplit", type=str, default="dev",
            choices=["train", "dev", "test"],
            help="Predict on this data split. Default 'dev'.")
    pred_parser.add_argument(
            "--datadir", type=str, default=None,
            help="""Path to directory containing .txt
                    files to run prediction on. If specified,
                    --sentences_dir must also be specified.""")
    pred_parser.add_argument(
            "--sentences_dir", type=str, default=None,
            help="""Path to directory containing sentence segmentations.
                    Only valid when --datadir is not None.""")
    pred_parser.add_argument(
            "--output_json", action="store_true", default=False,
            help="""If set, save json formatted predictions
                    to the predictions/ directory under the logdir.""")
    pred_parser.add_argument(
            "--output_token_masks", action="store_true", default=False,
            help="""If set, save tokens and stochastic masks to the
                    token_masks/ directory under the logdir.
                    Only valid for bert-rationale-model.""")

    return parser.parse_args()


def load_datamodule(config, stage, **override_kwargs):
    datamodule = load_datamodule_from_config(config, **override_kwargs)
    datamodule.setup(stage=stage)
    print(datamodule)
    print("Label Spec")
    print('  ' + str(datamodule.label_spec))
    return datamodule


def main(args):
    config = ExperimentConfig.from_yaml_file(args.config_file)
    logdir = config.logdir
    os.makedirs(logdir, exist_ok=True)

    curr_time = datetime.datetime.now()
    print(f"Start: {curr_time}")
    print(config)

    pl.seed_everything(config.random_seed, workers=True)

    run_kwargs = {
            "logdir": logdir,
            "quiet": args.quiet,
            }
    if args.command == "train":
        version = get_next_experiment_version(logdir, config.name)
        run_kwargs["version"] = version
        run_kwargs["save_last_epoch"] = args.save_last_epoch
        run_fn = run_train
    elif args.command == "continue":
        version = get_current_experiment_version(args.config_file)
        run_kwargs["version"] = version
        run_kwargs["save_last_epoch"] = args.save_last_epoch
        run_fn = run_continue_train
    elif args.command == "validate":
        run_kwargs["datasplit"] = args.datasplit
        version = get_current_experiment_version(args.config_file)
        run_kwargs["version"] = version
        run_fn = run_validate
    elif args.command == "predict":
        run_kwargs["datasplit"] = args.datasplit
        version = get_current_experiment_version(args.config_file)
        run_kwargs["version"] = version
        run_kwargs["datadir"] = args.datadir
        run_kwargs["sentences_dir"] = args.sentences_dir
        run_kwargs["output_json"] = args.output_json
        run_kwargs["output_token_masks"] = args.output_token_masks
        run_fn = run_predict
    else:
        raise argparse.ArgumentError(f"Unknown command {args.command}")
    run_fn(config, **run_kwargs)

    curr_time = datetime.datetime.now()
    print(f"End: {curr_time}")


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


def find_checkpoint(logdir, model_name, version, ckpt_glob="*.ckpt"):
    checkpoint_dir = os.path.join(
            logdir, model_name, f"version_{version}", "checkpoints")
    checkpoint_files = glob(os.path.join(checkpoint_dir, ckpt_glob))
    if len(checkpoint_files) == 0:
        raise OSError(f"No checkpoints found in {checkpoint_dir}")
    if len(checkpoint_files) > 1:
        raise OSError(f"Found multiple checkpoint files. Try a more specific glob? {checkpoint_files}")  # noqa
    checkpoint_file = checkpoint_files[0]
    hparams_file = os.path.join(checkpoint_dir, "../hparams.yaml")
    return checkpoint_file, hparams_file


def run_train(config, logdir="logs/", version=None,
              save_last_epoch=False, quiet=False):

    datamodule = load_datamodule(config, stage=None)

    version_dir = os.path.join(logdir, config.name, f"version_{version}")
    os.makedirs(version_dir, exist_ok=False)
    config.save_to_yaml(os.path.join(version_dir, "config.yaml"))

    model_class = MODEL_LOOKUP[config.model_name]
    model = model_class.from_config(config, datamodule)

    logger = TensorBoardLogger(
            save_dir=logdir, version=version, name=config.name)

    monitor_mode = "max"
    if "loss" in config.monitor:
        monitor_mode = "min"
    filename_fmt = f"{{epoch:02d}}-{{{config.monitor}:.2f}}"
    checkpoint_cb = ModelCheckpoint(
            monitor=config.monitor,
            mode=monitor_mode,
            filename=filename_fmt)

    available_gpus = min(1, torch.cuda.device_count())
    enable_progress_bar = not quiet
    trainer_kwargs = {"check_val_every_n_epoch": 1}
    # For CombinedDataModules with some samplers, the length can
    # change from epoch to epoch, which pytorch_lightning doesn't
    # really support. To get around this we just set the val
    # check interval to the smallest train epoch size.
    if isinstance(datamodule, CombinedDataModule):
        sampler = datamodule.train_dataloader().batch_sampler
        if hasattr(sampler, "step"):
            min_len = None
            for i in range(sampler.max_steps):
                sampler.step = i
                if min_len is None or len(sampler) < min_len:
                    min_len = len(sampler)
            trainer_kwargs = {"val_check_interval": min_len}
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
            **trainer_kwargs,
            )
    trainer.fit(model, datamodule=datamodule)
    if save_last_epoch is True:
        ckpt_dir = os.path.join(logdir, config.name,
                                f"version_{version}", "checkpoints")
        epoch = config.max_epochs - 1
        ckpt_path = os.path.join(ckpt_dir, f"last-train-epoch={epoch}.ckpt")
        trainer.save_checkpoint(ckpt_path)
        print(f"Final training epoch checkpointed at {ckpt_path}")


def run_continue_train(config, logdir="logs/", version=None,
                       save_last_epoch=False, quiet=False):
    datamodule = load_datamodule(config, stage=None)

    checkpoint_file, hparams_file = find_checkpoint(
            logdir, config.name, version, ckpt_glob="*epoch=*.ckpt")  # noqa
    model_class = MODEL_LOOKUP[config.model_name]
    model = model_class.load_from_checkpoint(
            checkpoint_path=checkpoint_file,
            hparams_file=hparams_file)

    logger = TensorBoardLogger(
            save_dir=logdir, version=version, name=config.name)

    checkpoint_cb = ModelCheckpoint(
            monitor="avg_macro_f1",
            mode="max",
            filename="{epoch:02d}-{avg_macro_f1:.2f}")

    available_gpus = min(1, torch.cuda.device_count())
    enable_progress_bar = not quiet
    ckpt_bn = os.path.basename(checkpoint_file)
    prev_epochs_match = re.match(r'epoch=([0-9]+)', ckpt_bn)
    if prev_epochs_match is not None:
        prev_epochs = int(prev_epochs_match.group(1))
    else:
        raise OSError("Couldn't find valid checkpoint file.")
    max_epochs = prev_epochs + 1 + config.max_epochs
    # There is a bug with deterministic indexing on the gpu
    #  in pytorch 1.10, so we have to turn it off.
    #  https://github.com/pytorch/pytorch/issues/61032
    trainer = pl.Trainer(
            logger=logger,
            max_epochs=max_epochs,
            gpus=available_gpus,
            gradient_clip_val=config.gradient_clip_val,
            deterministic=False,
            callbacks=[checkpoint_cb],
            enable_progress_bar=enable_progress_bar,
            log_every_n_steps=32,
            )

    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_file)
    if save_last_epoch is True:
        os.remove(checkpoint_file)
        ckpt_dir = os.path.join(logdir, config.name,
                                f"version_{version}", "checkpoints")
        epoch = max_epochs - 1
        ckpt_path = os.path.join(ckpt_dir, f"last-train-epoch={epoch}.ckpt")
        trainer.save_checkpoint(ckpt_path)
        print(f"Final training epoch checkpointed at {ckpt_path}")


def run_validate(config, datasplit="dev",
                 logdir="logs/", version=None, quiet=False):

    datamodule = load_datamodule(config, stage=None)

    checkpoint_file, hparams_file = find_checkpoint(
            logdir, config.name, version, ckpt_glob="epoch*.ckpt")

    model_class = MODEL_LOOKUP[config.model_name]
    model = model_class.load_from_checkpoint(
            checkpoint_path=checkpoint_file,
            hparams_file=hparams_file)
    model.eval()

    available_gpus = min(1, torch.cuda.device_count())
    enable_progress_bar = not quiet
    trainer = pl.Trainer(
            logger=False,  # Disable tensorboard logging
            gpus=available_gpus,
            enable_progress_bar=enable_progress_bar
            )
    if datasplit == "train":
        val_dataloader_fn = datamodule.train_dataloader
    elif datasplit == "dev":
        val_dataloader_fn = datamodule.val_dataloader
    elif datasplit == "test":
        if datamodule.test_dataloader() is None:
            raise OSError("No test data found. Aborting.")
        val_dataloader_fn = datamodule.test_dataloader
    else:
        raise ValueError(f"Unknown validation data split '{datasplit}'")
    val_dataloader = val_dataloader_fn()
    results = trainer.validate(
        model, dataloaders=val_dataloader, verbose=False)[0]
    md = format_results_as_markdown_table(results)
    print(md)


def run_predict(config, datasplit="dev",
                datadir=None, sentences_dir=None,
                logdir="logs/", version=None, quiet=False,
                output_json=False, output_token_masks=False):

    stage = None
    data_kwargs = {}
    if datadir is not None:
        if sentences_dir is None:
            raise ValueError("Must specify either both or neither of --datadir, --sentences_dir")  # noqa
        # This tells the dataloader not to look for gold labels.
        ann_glob = os.path.join(datadir, "*.ann")
        assert len(ann_glob) > 0, f"No .ann files found at {ann_glob}"
        stage = "predict"
        data_kwargs = {"data_dir": datadir,
                       "sentences_dir": sentences_dir,
                       "auxiliary_data": {}}
    datamodule = load_datamodule(config, stage=stage, **data_kwargs)

    checkpoint_file, hparams_file = find_checkpoint(
            logdir, config.name, version, ckpt_glob="epoch*.ckpt")

    model_class = MODEL_LOOKUP[config.model_name]
    model = model_class.load_from_checkpoint(
            checkpoint_path=checkpoint_file,
            hparams_file=hparams_file)
    model.eval()

    available_gpus = min(1, torch.cuda.device_count())
    enable_progress_bar = not quiet
    trainer = pl.Trainer(
            logger=False,  # Disable tensorboard logging
            gpus=available_gpus,
            enable_progress_bar=enable_progress_bar
            )
    if stage == "predict":
        pred_dataloader_fn = datamodule.predict_dataloader
    else:
        if datasplit == "train":
            pred_dataloader_fn = datamodule.train_dataloader
        elif datasplit == "dev":
            pred_dataloader_fn = datamodule.val_dataloader
        elif datasplit == "test":
            if datamodule.test_dataloader() is None:
                raise OSError("No test data found. Aborting.")
            pred_dataloader_fn = datamodule.test_dataloader

    pred_dataloader_kwargs = {}
    if isinstance(datamodule, CombinedDataModule):
        pred_dataloader_kwargs = {"predicting": True}
    pred_dataloader = pred_dataloader_fn(**pred_dataloader_kwargs)
    preds = trainer.predict(model, dataloaders=pred_dataloader)

    # Output to brat
    anns_by_datatset = batched_predictions_to_brat(preds, datamodule)
    for (dataset, anns) in anns_by_datatset.items():
        preds_dir = os.path.join(logdir, config.name, f"version_{version}",
                                 "predictions", dataset, "brat", datasplit)
        if os.path.isdir(preds_dir):
            warnings.warn("brat predictions directory already exists at {preds_dir}. Skipping...")  # noqa
            continue
        os.makedirs(preds_dir, exist_ok=False)
        for doc_anns in anns:
            doc_anns.save_brat(preds_dir)

    if output_json is True:
        if stage == "predict":
            # We don't have gold labels so this won't work.
            raise ValueError("--output_json requires gold labels, which were not loaded.")  ## noqa

        preds_by_dataset = batched_predictions_to_json(preds, datamodule)
        for (dataset, preds_by_task) in preds_by_dataset.items():
            preds_dir = os.path.join(logdir, config.name, f"version_{version}",
                                     "predictions", dataset, "json", datasplit)
            if os.path.isdir(preds_dir):
                warnings.warn("JSON predictions directory already exists at {preds_dir}. Skipping...")  # noqa
                continue
            os.makedirs(preds_dir, exist_ok=False)
            for (task, preds) in preds_by_task.items():
                outpath = os.path.join(preds_dir, f"{task}.jsonl")
                with open(outpath, 'w') as outF:
                    for datum in preds:
                        json.dump(datum, outF)
                        outF.write('\n')

    if output_token_masks is True:
        if config.model_name != "bert-rationale-classifier":
            raise ValueError("--output_token_masks only compatible with bert-rationale-classifier.")  # noqa
        raise NotImplementedError("Still working on this...")
        mask_dir = os.path.join(
                logdir, config.name, f"version_{version}",
                "predictions", dataset, "token_masks", datasplit)
        os.makedirs(mask_dir, exist_ok=False)
        masked_by_task = batched_predictions_to_masked_tokens(
                preds, datamodule)
        for (task, masked_examples) in masked_by_task.items():
            outpath = os.path.join(mask_dir, f"{task}.jsonl")
            with open(outpath, 'w') as outF:
                for datum in masked_examples:
                    json.dump(datum, outF)
                    outF.write('\n')


def maybe_split_results_by_dataset(results_dict):
    # TODO: make metric names follow {task}_{avg_fn}_{metric}
    #       to simplify this code.
    #       Even better, always output results keyed by dataset
    did_split = False
    new_results = defaultdict(dict)
    for (metric_name, val) in results_dict.items():
        if ':' in metric_name:
            did_split = True
            tmp = metric_name.split('_')
            metric_idx = [i for i in range(len(tmp)) if ':' in tmp[i]][0]
            dataset, metric_str = tmp[metric_idx].split(':')
            tmp[metric_idx] = metric_str
            new_results[dataset]['_'.join(tmp)] = val
    return did_split, new_results


def format_results_as_markdown_table(results):
    """
    ### Dataset Name
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
            "Temporality": "Temp",
            "Assertion": "Assert",
            }
    did_split, results_by_dataset = maybe_split_results_by_dataset(results)
    if did_split is True:
        total_table = ''
        for (dataset, results) in results_by_dataset.items():
            dataset_table = format_results_as_markdown_table(results)
            dataset_table = f"\n### {dataset}\n" + dataset_table
            total_table += dataset_table
        return total_table

    tasks = set()
    for key in results.keys():
        if key.startswith("micro") or key.startswith("macro"):
            avg_fn, task, metric = key.split('_')
            tasks.add(task)
    tasks = sorted(tasks)

    # Header
    table = '|'
    for task in tasks:
        try:
            task_str = task_abbrevs[task]
        except KeyError:
            task_str = task[:6]
        table += f" {task_str: <7} | {'P': <5} | {'R': <5} | {'F1': <5} |"

    # Separator
    table += "\n|"
    for task in tasks:
        table += "---------|-------|-------|-------|"

    # Results
    for avg_fn in ["micro", "macro"]:
        table += "\n|"
        for task in tasks:
            try:
                p = results[f"{avg_fn}_{task}_precision"]
                r = results[f"{avg_fn}_{task}_recall"]
                f = results[f"{avg_fn}_{task}_F1"]
                table += f" {avg_fn: <7} | {p:.3f} | {r:.3f} | {f:.3f} |"
            except KeyError:
                # probs that an auxiliary dataset doesn't have a dev split.
                continue
    table += '\n'
    return table


# TODO: make this work in the multi-dataset setup.
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


def batched_predictions_to_json(preds, datamodule):
    """
    Used by utils/predviewer.py
    """
    preds_by_dataset_and_task = defaultdict(lambda: defaultdict(list))
    for batch in preds:
        tmp = [getattr(datamodule, split, None) for split in
               ["train", "val", "test", "predict"]
               if getattr(datamodule, split, None) is not None]
        default_dataset_name = tmp[0].name
        dataset = default_dataset_name
        for (i, docid) in enumerate(batch["docids"]):
            for (task, preds) in batch["predictions"].items():
                base_task = task
                if isinstance(datamodule, CombinedDataModule):
                    dataset_ = datamodule.get_dataset_from_task(task).name
                    # so we don't output the dataset name
                    base_task = task.split(':')[-1]
                    if dataset == default_dataset_name:
                        dataset = dataset_
                    else:
                        assert dataset_ == dataset, f"Datasets should not differ within a docid! Got {dataset} and {dataset_} for docid {docid}"  # noqa
                datum = {}
                datum["docid"] = docid
                input_ids = batch["input_ids"][i]
                tokens = datamodule.tokenizer.convert_ids_to_tokens(
                        input_ids, skip_special_tokens=True)
                tokens = [tok for (tok, tid) in zip(tokens, input_ids) if tid != 0]  # noqa
                datum["tokens"] = tokens
                enc_pred = preds[i].int().item()
                datum["prediction"] = datamodule.inverse_transform(
                        task, [enc_pred])[0]
                enc_lab = batch["labels"][task][i].int().item()
                datum["label"] = datamodule.inverse_transform(
                        task, [enc_lab])[0]
                preds_by_dataset_and_task[dataset][base_task].append(datum)
    return preds_by_dataset_and_task


def batched_predictions_to_brat(preds, datamodule):
    """
    Create a BratAnnotations instance for each grouped of predictions,
    grouped by dataset and doc_id.
    """
    events_by_dataset_and_docid = defaultdict(lambda: defaultdict(list))
    for batch in preds:
        # TODO: When using a CombinedDataModule, tasks may be split across
        #       batches for a given example. Group by all docids first,
        #       then iterate over them to create the output documents.
        #   This is a bit of side-case, however, when we load two tasks
        #       from the same dataset as separate datasets (i.e., one is
        #       loaded under auxiliary_data.
        tmp = [getattr(datamodule, split, None) for split in
               ["train", "val", "test", "predict"]
               if getattr(datamodule, split, None) is not None]
        default_dataset_name = tmp[0].name
        dataset = default_dataset_name
        task_warning_seen = False
        for (i, docid) in enumerate(batch["docids"]):
            attrs = {}
            for (task, task_preds) in batch["predictions"].items():
                encoded_pred = task_preds[i].int().item()
                # TODO: This try - except is a hack. Fix it!
                try:
                    decoded_pred = datamodule.inverse_transform(
                        task, encoded_pred)
                except KeyError:
                    msg = f"{task} not supported by datamodule {datamodule.name}"  # noqa
                    if task_warning_seen is False:
                        warnings.warn(msg)
                        task_warning_seen = True
                    continue
                if isinstance(datamodule, CombinedDataModule):
                    dataset_ = datamodule.get_dataset_from_task(task).name
                    # so we don't output the datset name in the brat
                    task = task.split(':')[-1]
                    if dataset == default_dataset_name:
                        dataset = dataset_
                    else:
                        assert dataset_ == dataset, f"Datasets should not differ within a docid! Got {dataset} and {dataset_} for docid {docid}"  # noqa
                num_attrs = len(
                    [attr for e in events_by_dataset_and_docid[dataset][docid]
                     for attr in e.attributes]
                ) + len(attrs)
                aid = f"A{num_attrs}"
                attr = br.Attribute(_id=aid, _type=task,
                                    value=decoded_pred, reference=None)
                attrs[task] = attr

            # Reconstruct original character offsets
            start, end = batch["entity_char_spans"][i]
            entity_text = batch["texts"][i][start:end]
            # There's one entity mention in n2c2 2022 that spans a newline
            #  so we'll fix that here.
            entity_text = entity_text.replace('\n', ' ')
            start += batch["char_offsets"][i]
            end += batch["char_offsets"][i]

            # Remove the '@' entity markers if used.
            if datamodule.entity_markers is not None:
                start_marker, end_marker = datamodule.entity_markers
                entity_text = entity_text.replace(start_marker, '')
                entity_text = entity_text.replace(end_marker, '')
                end -= sum([len(em) for em in datamodule.entity_markers])

            num_spans = len(
                set(
                    [(e.span.start_index, e.span.end_index)
                     for e in events_by_dataset_and_docid[dataset][docid]]
                )
            )
            sid = f"T{num_spans}"
            span = br.Span(_id=sid, _type="Disposition", start_index=start,
                           end_index=end, text=entity_text)

            src_file_str = f"{docid}.ann"
            if dataset is None:
                dataset = datamodule.name
            eid = f"E{len(events_by_dataset_and_docid[dataset][docid])}"
            event = br.Event(_id=eid, _type="Disposition", span=span,
                             attributes=attrs, _source_file=src_file_str)
            events_by_dataset_and_docid[dataset][docid].append(event)

    anns_by_dataset = {}
    for dataset in events_by_dataset_and_docid.keys():
        anns_by_dataset[dataset] = [
            br.BratAnnotations.from_events(events)
            for events in events_by_dataset_and_docid[dataset].values()]
    return anns_by_dataset


if __name__ == "__main__":
    args = parse_args()
    main(args)
