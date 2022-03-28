import argparse
import colorama
import warnings

import torch
import pytorch_lightning as pl

from config import ExperimentConfig
from data.n2c2 import n2c2DataModule
from models import BertMultiHeadedSequenceClassifier
from models.layers import TokenMask, EntityPooler


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
        except AssertionError as e:
            res_str = "Failed"
            color = colorama.Fore.RED
            print(e)
        print(color + f" [{res_str}]")
    return wrapper


def run(config_file):
    tmp_config = ExperimentConfig.from_yaml_file(config_file)
    # Generate default config and override options
    config = ExperimentConfig()
    config.data_dir = tmp_config.data_dir
    config.sentences_dir = tmp_config.sentences_dir
    config.model_name = "bert-sequence-classifier"
    config.bert_model_name_or_path = "bert-base-uncased"
    config.max_seq_length = 300
    del tmp_config
    pl.seed_everything(config.random_seed)
    print(config)

    test_mask_hidden(config)
    test_mask_hidden_marked(config)
    test_pool_entity_embeddings_max(config)
    test_pool_entity_embeddings_mean(config)
    test_pool_entity_embeddings_first(config)
    test_pool_entity_embeddings_last(config)
    test_pool_entity_embeddings_first_last(config)


@test_logger
def test_mask_hidden(config):
    data_kwargs = {
        "mark_entities": False
    }
    datamodule = n2c2DataModule.from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    assert datamodule.mark_entities is False
    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "first",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    assert model.use_entity_spans is True
    assert model.entity_pool_fn == "first"

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
def test_mask_hidden_marked(config):
    data_kwargs = {
        "mark_entities": True,
    }
    datamodule = n2c2DataModule.from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    assert datamodule.mark_entities is True

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "first-last",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    assert model.use_entity_spans is True
    assert model.entity_pool_fn == "first-last"

    start_marker = datamodule.train.START_ENTITY_MARKER
    end_marker = datamodule.train.END_ENTITY_MARKER
    start_id = datamodule.tokenizer.convert_tokens_to_ids([start_marker])[0]
    end_id = datamodule.tokenizer.convert_tokens_to_ids([end_marker])[0]
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
            assert masked_ids[0] == start_id and masked_ids[-1] == end_id


@test_logger
def test_pool_entity_embeddings_max(config):
    data_kwargs = {
        "mark_entities": False,
    }
    datamodule = n2c2DataModule.from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    assert datamodule.mark_entities is False

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "max",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    assert model.use_entity_spans is True
    assert model.entity_pool_fn == "max"

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
def test_pool_entity_embeddings_mean(config):
    data_kwargs = {
        "mark_entities": False,
    }
    datamodule = n2c2DataModule.from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    assert datamodule.mark_entities is False

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "mean",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    assert model.use_entity_spans is True
    assert model.entity_pool_fn == "mean"

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
def test_pool_entity_embeddings_first(config):
    data_kwargs = {
        "mark_entities": False,
    }
    datamodule = n2c2DataModule.from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    assert datamodule.mark_entities is False

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "first",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    assert model.use_entity_spans is True
    assert model.entity_pool_fn == "first"

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
def test_pool_entity_embeddings_last(config):
    data_kwargs = {
        "mark_entities": False,
    }
    datamodule = n2c2DataModule.from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    assert datamodule.mark_entities is False

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "last",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    assert model.use_entity_spans is True
    assert model.entity_pool_fn == "last"

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
def test_pool_entity_embeddings_first_last(config):
    data_kwargs = {
        "mark_entities": False,
    }
    datamodule = n2c2DataModule.from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    assert datamodule.mark_entities is False

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "first-last",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    assert model.use_entity_spans is True
    assert model.entity_pool_fn == "first-last"

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
