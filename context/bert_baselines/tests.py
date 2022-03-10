import argparse
import colorama

import torch
import pytorch_lightning as pl

from config import ExperimentConfig
from data import n2c2ContextDataModule
from model import BertMultiHeadedSequenceClassifier
from layers import TokenMask, EntityPooler


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

    datamodule = n2c2ContextDataModule(
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
    test_mask_hidden_marked(config, datamodule)
    test_pool_entity_embeddings_max(config, datamodule)
    test_pool_entity_embeddings_mean(config, datamodule)
    test_pool_entity_embeddings_first(config, datamodule)
    test_pool_entity_embeddings_last(config, datamodule)
    test_pool_entity_embeddings_first_last(config, datamodule)


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

    masker = TokenMask()
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            h = torch.randn(config.batch_size, config.max_seq_length,
                            model.bert_config.hidden_size)
            offset_mapping = batch["encodings"]["offset_mapping"]
            entity_spans = batch["entity_spans"]
            try:
                masked, token_mask = masker(
                    h, offset_mapping, entity_spans)
            # mask_hidden raises ValueError("Entity span not found!")
            except ValueError:
                raise AssertionError("Entity span not found.")


@test_logger
def test_mask_hidden_marked(config, datamodule):
    datamodule = n2c2ContextDataModule(
        config.data_dir,
        config.sentences_dir,
        batch_size=1,
        bert_model_name_or_path=config.bert_model_name_or_path,
        tasks_to_load=config.tasks_to_load,
        max_seq_length=config.max_seq_length,
        window_size=config.window_size,
        max_train_examples=None,
        mark_entities=True)
    datamodule.setup()

    model = BertMultiHeadedSequenceClassifier(
        config.bert_model_name_or_path,
        label_spec=datamodule.label_spec,
        freeze_pretrained=True,
        use_entity_spans=True,
        entity_pool_fn="max",
        lr=config.lr,
        weight_decay=config.weight_decay)

    at_id = datamodule.tokenizer.convert_tokens_to_ids(['@'])[0]
    masker = TokenMask()
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            h = torch.randn(1, config.max_seq_length,
                            model.bert_config.hidden_size)
            offset_mapping = batch["encodings"]["offset_mapping"]
            entity_spans = batch["entity_spans"]
            try:
                masked, token_mask = masker(
                    h, offset_mapping, entity_spans)
            # mask_hidden raises ValueError("Entity span not found!")
            except ValueError:
                raise AssertionError("Entity span not found.")
            masked_ids = batch["encodings"]["input_ids"][token_mask[:, :, 0]]
            assert masked_ids[0] == at_id and masked_ids[-1] == at_id


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

    masker = TokenMask()
    insize = outsize = model.bert_config.hidden_size
    pooler = EntityPooler(insize, outsize, "max")
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            h = torch.randn(config.batch_size, config.max_seq_length,
                            model.bert_config.hidden_size)
            offset_mapping = batch["encodings"]["offset_mapping"]
            entity_spans = batch["entity_spans"]
            masked, token_mask = masker(h, offset_mapping, entity_spans)
            pooled = pooler.pooler(masked, token_mask)
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

    masker = TokenMask()
    insize = outsize = model.bert_config.hidden_size
    pooler = EntityPooler(insize, outsize, "mean")
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            h = torch.randn(config.batch_size, config.max_seq_length,
                            model.bert_config.hidden_size)
            offset_mapping = batch["encodings"]["offset_mapping"]
            entity_spans = batch["entity_spans"]
            masked, token_mask = masker(h, offset_mapping, entity_spans)
            pooled = pooler.pooler(masked, token_mask)
            assert not torch.isnan(pooled).all()
            assert not torch.isinf(pooled).all()


@test_logger
def test_pool_entity_embeddings_first(config, datamodule):
    model = BertMultiHeadedSequenceClassifier(
        config.bert_model_name_or_path,
        label_spec=datamodule.label_spec,
        freeze_pretrained=True,
        use_entity_spans=True,
        entity_pool_fn="first",
        lr=config.lr,
        weight_decay=config.weight_decay)

    masker = TokenMask()
    insize = outsize = model.bert_config.hidden_size
    pooler = EntityPooler(insize, outsize, "first")

    h = torch.arange(config.max_seq_length).unsqueeze(-1).expand(
        -1, model.bert_config.hidden_size)
    h = torch.stack([h] * config.batch_size)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            offset_mapping = batch["encodings"]["offset_mapping"]
            entity_spans = batch["entity_spans"]
            masked, token_mask = masker(h, offset_mapping, entity_spans)
            pooled = pooler.pooler(masked, token_mask)
            assert not torch.isnan(pooled).all()
            assert not torch.isinf(pooled).all()


@test_logger
def test_pool_entity_embeddings_last(config, datamodule):
    model = BertMultiHeadedSequenceClassifier(
        config.bert_model_name_or_path,
        label_spec=datamodule.label_spec,
        freeze_pretrained=True,
        use_entity_spans=True,
        entity_pool_fn="last",
        lr=config.lr,
        weight_decay=config.weight_decay)

    masker = TokenMask()
    insize = outsize = model.bert_config.hidden_size
    pooler = EntityPooler(insize, outsize, "last")

    h = torch.arange(config.max_seq_length).unsqueeze(-1).expand(
        -1, model.bert_config.hidden_size)
    h = torch.stack([h] * config.batch_size)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            offset_mapping = batch["encodings"]["offset_mapping"]
            entity_spans = batch["entity_spans"]
            masked, token_mask = masker(h, offset_mapping, entity_spans)
            pooled = pooler.pooler(masked, token_mask)
            assert not torch.isnan(pooled).all()
            assert not torch.isinf(pooled).all()


@test_logger
def test_pool_entity_embeddings_first_last(config, datamodule):
    model = BertMultiHeadedSequenceClassifier(
        config.bert_model_name_or_path,
        label_spec=datamodule.label_spec,
        freeze_pretrained=True,
        use_entity_spans=True,
        entity_pool_fn="first-last",
        lr=config.lr,
        weight_decay=config.weight_decay)

    masker = TokenMask()
    insize = outsize = model.bert_config.hidden_size
    pooler = EntityPooler(insize, outsize, "first-last")

    h = torch.arange(config.max_seq_length).unsqueeze(-1).expand(
        -1, model.bert_config.hidden_size)
    h = torch.stack([h] * config.batch_size)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            offset_mapping = batch["encodings"]["offset_mapping"]
            entity_spans = batch["entity_spans"]
            masked, token_mask = masker(h, offset_mapping, entity_spans)
            pooled = pooler.pooler(masked, token_mask)
            assert not torch.isnan(pooled).all()
            assert not torch.isinf(pooled).all()


if __name__ == "__main__":
    args = parse_args()
    run(args.config_file)
