import timeit
import argparse
import colorama
import warnings
import logging

import torch
import numpy as np
import pytorch_lightning as pl

from src.config import ExperimentConfig
from src.data import load_datamodule_from_config, combined
from src.models import BertMultiHeadedSequenceClassifier
from src.models.layers import TokenMask, EntityPooler


logging.getLogger("pytorch_lightning.utilities.seed").setLevel(logging.WARNING)

colorama.init(autoreset=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    return parser.parse_args()


def test_logger(func):
    def wrapper(*args, **kwargs):
        print(func.__name__, end='', flush=True)
        try:
            func(*args, **kwargs)
            res_str = "Passed"
            color = colorama.Fore.GREEN
        except AssertionError:
            res_str = "Failed"
            color = colorama.Fore.RED
        print(color + f" [{res_str}]", flush=True)
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

    test_mask_hidden(config)
    test_mask_hidden_marked(config)
    test_pool_entity_embeddings_max(config)
    test_pool_entity_embeddings_mean(config)
    test_pool_entity_embeddings_first(config)
    test_pool_entity_embeddings_last(config)
    test_pool_entity_embeddings_first_last(config)
    test_scheduled_dataset_sampler(config)
    test_stickland_murray_dataset_sampler(config)
    test_levitate_encodings(config)


@test_logger
def test_mask_hidden(config):
    pl.utilities.seed.reset_seed()
    data_kwargs = {
        "mark_entities": False
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
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
    pl.utilities.seed.reset_seed()
    data_kwargs = {
        "mark_entities": True,
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
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
    pl.utilities.seed.reset_seed()
    data_kwargs = {
        "mark_entities": False,
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
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
    pl.utilities.seed.reset_seed()
    data_kwargs = {
        "mark_entities": False,
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
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
    pl.utilities.seed.reset_seed()
    data_kwargs = {
        "mark_entities": False,
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
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
    pl.utilities.seed.reset_seed()
    data_kwargs = {
        "mark_entities": False,
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
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
    pl.utilities.seed.reset_seed()
    data_kwargs = {
        "mark_entities": False,
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
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


@test_logger
def test_scheduled_dataset_sampler(config):
    pl.utilities.seed.reset_seed()
    data_kwargs = {
        "batch_size": 16,
        "max_train_examples": None,
        "auxiliary_data": {
            "n2c2Assertion": {
                "dataset_name": "n2c2Assertion",
                "data_dir": "/home/jav/Documents/Projects/n2c2_2022/n2c2-track1/auxiliary_data/n2c2_2010_concept_assertion_relation/combined/ast_brat/",  # noqa
                "sentences_dir": "/home/jav/Documents/Projects/n2c2_2022/n2c2-track1/auxiliary_data/n2c2_2010_concept_assertion_relation/combined/segmented/",  # noqa
                "max_train_examples": None,
                "tasks_to_load": ["Assertion"],
                "window_size": 0,
            },
        },
        "dataset_sample_strategy": "scheduled",
        "dataset_sampler_kwargs": {
            "weights": [0.9, 0.1],
            "exhaust_all": False,
            "num_cycles": 1,
            "invert": False,
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore type warnings
        datamodule = load_datamodule_from_config(config, **data_kwargs,
                                                 errors="warn")
        # Ignore "No test set found"
        datamodule.setup()
    assert isinstance(datamodule, combined.CombinedDataModule)

    sampler = datamodule.train_dataloader().batch_sampler
    assert isinstance(sampler, combined.ScheduledWeightedSampler)

    ntrials = 10
    sampler_lens = np.zeros(sampler.max_steps)
    batch_lens = np.zeros((ntrials, sampler.max_steps))
    for i in range(ntrials):
        sampler.reset()
        for j in range(sampler.max_steps):
            for (k, batch) in enumerate(sampler):
                # Could be less than if we exhaust a dataset this step
                assert len(batch) <= sampler.batch_size
                assert len(batch) <= datamodule.batch_size
                if k == 0:
                    sampler_lens[j] = sampler.target_samples_this_epoch
                batch_lens[i, j] += 1

    batch_matches_sampler = (batch_lens == sampler_lens).sum(axis=0)
    assert (batch_matches_sampler == ntrials).all()

    sampler.reset()
    dataset_lens = [len(ds) for ds in sampler.dataset.datasets]
    gold_lens = np.array(
        [sampler._estimate_length(dataset_lens, sampler.get_dataset_probs(s))
         for s in range(sampler.max_steps)]
    )
    assert (gold_lens == sampler_lens).all()


@test_logger
def test_stickland_murray_dataset_sampler(config):
    pl.utilities.seed.reset_seed()
    data_kwargs = {
        "batch_size": 16,
        "max_train_examples": None,
        "auxiliary_data": {
            "n2c2Assertion": {
                "dataset_name": "n2c2Assertion",
                "data_dir": "/home/jav/Documents/Projects/n2c2_2022/n2c2-track1/auxiliary_data/n2c2_2010_concept_assertion_relation/combined/ast_brat/",  # noqa
                "sentences_dir": "/home/jav/Documents/Projects/n2c2_2022/n2c2-track1/auxiliary_data/n2c2_2010_concept_assertion_relation/combined/segmented/",  # noqa
                "max_train_examples": 200,
                "tasks_to_load": ["Assertion"],
                "window_size": 0,
            },
        },
        "dataset_sample_strategy": "stickland-murray",
        "dataset_sampler_kwargs": {
            "max_steps": 10,
            "exhaust_all": False,
            "anneal_rate": 0.8,
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore type warnings
        datamodule = load_datamodule_from_config(
            config, **data_kwargs, errors="warn")
        # Ignore "No test set found"
        datamodule.setup()
    assert isinstance(datamodule, combined.CombinedDataModule)

    sampler = datamodule.train_dataloader().batch_sampler
    assert isinstance(sampler, combined.SticklandMurraySampler)

    ntrials = 10
    start_probs = None
    end_probs = None
    sampler_lens = np.zeros(sampler.max_steps)
    batch_lens = np.zeros((ntrials, sampler.max_steps))
    for i in range(ntrials):
        sampler.reset()
        for j in range(sampler.max_steps):
            for (k, batch) in enumerate(sampler):
                # Could be less than if we exhaust a dataset this step
                assert len(batch) <= sampler.batch_size
                assert len(batch) <= datamodule.batch_size
                if i == 0 and k == 0:
                    sampler_lens[j] = sampler.target_samples_this_epoch
                    start_probs = sampler.get_dataset_probs()
                batch_lens[i, j] += 1
        end_probs = sampler.get_dataset_probs()
        # The probabilities should get closer together
        start_diff = np.subtract(*start_probs)
        if len(end_probs) == 1:
            end_diff = 0.
        else:
            end_diff = np.subtract(*end_probs)
        assert start_diff > end_diff

    batch_matches_sampler = (batch_lens == sampler_lens).sum(axis=0)
    assert (batch_matches_sampler == ntrials).all()

    sampler.reset()
    dataset_lens = [len(ds) for ds in sampler.dataset.datasets]
    gold_lens = np.array(
        [sampler._estimate_length(dataset_lens, sampler.get_dataset_probs(s))
         for s in range(sampler.max_steps)]
    )
    assert (gold_lens == sampler_lens).all()


def test_levitate_encodings(config, i=0):
    pl.utilities.seed.reset_seed()
    data_kwargs = {
        "mark_entities": True,
        "batch_size": 2,
        "max_seq_length": 75,
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    for (batch_idx, batch) in enumerate(datamodule.val_dataloader()):
        if batch_idx == i:
            break
    start = timeit.default_timer()
    encodings, marker_spans = datamodule.levitate_encodings(
        batch["encodings"], batch["entity_spans"])
    end = timeit.default_timer()
    print(f"levitate_encodings(): {end - start}s")
    # TODO: come up with some actual tests.
    return encodings, marker_spans


if __name__ == "__main__":
    args = parse_args()
    run(args.config_file)
