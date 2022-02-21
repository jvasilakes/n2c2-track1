import argparse
import colorama

import torch
import pytorch_lightning as pl

from config import ExperimentConfig
from data import n2c2SentencesDataModule
from model import BertMultiHeadedSequenceClassifier


colorama.init(autoreset=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    return parser.parse_args()


def test_logger(func):
    def wrapper(*args, **kwargs):
        print(func.__name__, end='')
        try:
            func(*args, **kwargs)
            res_str = "Passed"
            color = colorama.Fore.GREEN
        except AssertionError:
            res_str = "Failed"
            color = colorama.Fore.RED
        print(color + f" [{res_str}]")
    return wrapper


def run(config_file):
    config = ExperimentConfig.from_yaml_file(config_file)
    pl.seed_everything(config.random_seed)

    datamodule = n2c2SentencesDataModule(
        config.data_dir,
        config.sentences_dir,
        batch_size=config.batch_size,
        bert_model_name_or_path=config.bert_model_name_or_path,
        tasks_to_load=config.tasks_to_load,
        max_seq_length=config.max_seq_length,
        window_size=config.window_size,
        max_train_examples=None)
    datamodule.setup()

    test_mask_hidden(config, datamodule)
    test_pool_entity_embeddings_max(config, datamodule)
    test_pool_entity_embeddings_mean(config, datamodule)


@test_logger
def test_mask_hidden(config, datamodule):
    model = BertMultiHeadedSequenceClassifier(
        config.bert_model_name_or_path,
        label_spec=datamodule.label_spec,
        freeze_pretrained=True,
        use_entity_spans=True,
        entity_pool_fn="max",
        lr=config.lr,
        weight_decay=config.weight_decay)

    h = torch.randn(config.batch_size, config.max_seq_length,
                    model.bert_config.hidden_size)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            offset_mapping = batch["encodings"]["offset_mapping"]
            entity_spans = batch["entity_spans"]
            try:
                masked, token_mask = model.mask_hidden(
                    h, offset_mapping, entity_spans)
            except ValueError:
                raise AssertionError()


@test_logger
def test_pool_entity_embeddings_max(config, datamodule):
    model = BertMultiHeadedSequenceClassifier(
        config.bert_model_name_or_path,
        label_spec=datamodule.label_spec,
        freeze_pretrained=True,
        use_entity_spans=True,
        entity_pool_fn="max",
        lr=config.lr,
        weight_decay=config.weight_decay)

    h = torch.randn(config.batch_size, config.max_seq_length,
                    model.bert_config.hidden_size)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            offset_mapping = batch["encodings"]["offset_mapping"]
            entity_spans = batch["entity_spans"]
            masked, token_mask = model.mask_hidden(
                h, offset_mapping, entity_spans)
            pooled = model.pool_entity_embeddings(masked, token_mask)
            assert (pooled == torch.tensor(0.)).any() == torch.tensor(False)


@test_logger
def test_pool_entity_embeddings_mean(config, datamodule):
    model = BertMultiHeadedSequenceClassifier(
        config.bert_model_name_or_path,
        label_spec=datamodule.label_spec,
        freeze_pretrained=True,
        use_entity_spans=True,
        entity_pool_fn="mean",
        lr=config.lr,
        weight_decay=config.weight_decay)

    h = torch.randn(config.batch_size, config.max_seq_length,
                    model.bert_config.hidden_size)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            offset_mapping = batch["encodings"]["offset_mapping"]
            entity_spans = batch["entity_spans"]
            masked, token_mask = model.mask_hidden(
                h, offset_mapping, entity_spans)
            pooled = model.pool_entity_embeddings(masked, token_mask)
            assert not torch.isnan(pooled).all()
            assert not torch.isinf(pooled).all()


# TODO: Test weight sampling.
@test_logger
def test_compute_sample_weights():
    raise NotImplementedError()


if __name__ == "__main__":
    args = parse_args()
    run(args.config_file)
