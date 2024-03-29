import os
import re
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
from src.models.layers import TokenEmbeddingPooler


os.environ["TOKENIZERS_PARALLELISM"] = "true"

logging.getLogger("pytorch_lightning.utilities.seed").setLevel(logging.WARNING)

colorama.init(autoreset=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_home", type=str,
                        help="Absolute path to n2c2-track1")
    return parser.parse_args()


def test_logger(func):
    def wrapper(*args, **kwargs):
        print(func.__name__, end='', flush=True)
        pl.seed_everything(0)
        error = None
        try:
            func(*args, **kwargs)
            res_str = "Passed"
            color = colorama.Fore.GREEN
        except AssertionError as e:
            error = e
            res_str = "Failed"
            color = colorama.Fore.RED
        print(color + f" [{res_str}]", flush=True)
        if error is not None:
            color = colorama.Fore.RED
            print(color + f"  ⤷ {error}", flush=True)
    return wrapper


def run(project_home):
    # Generate default config and override options
    config = ExperimentConfig()
    config.data_dir = f"{project_home}/n2c2Track1TrainingData-v3/data_v3/"
    config.sentences_dir = f"{project_home}/n2c2Track1TrainingData-v3/segmented"  # noqa
    config.model_name = "bert-sequence-classifier"
    config.bert_model_name_or_path = "bert-base-uncased"
    config.max_seq_length = 300
    config.batch_size = 2
    config.use_levitated_markers = False

    # TODO: This test hangs if not run first. I don't know why.
    test_pool_entity_embeddings_first(config)
    test_pool_entity_embeddings_max(config)
    test_pool_entity_embeddings_mean(config)
    test_pool_entity_embeddings_last(config)
    test_pool_entity_embeddings_first_last(config)
    test_get_token_indices_from_char_span(config)
    test_mask_hidden(config)
    test_mask_hidden_marked(config)
    test_scheduled_dataset_sampler(config, project_home)
    test_stickland_murray_dataset_sampler(config, project_home)
    test_levitate_encodings(config)


@test_logger
def test_get_token_indices_from_char_span(config):
    data_kwargs = {
        "entity_markers": '@',
        "batch_size": 1,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore changing type of config parameters
        datamodule = load_datamodule_from_config(config, **data_kwargs)
        # Ignore "No test set found"
        datamodule.setup()
    msg = "datamodule.entity_markers expected to be ('@', '@')"
    assert datamodule.entity_markers == ('@', '@'), msg

    for ds in [datamodule.train, datamodule.val]:
        for example in ds:
            encoded = datamodule.tokenizer(
                    example["text"], truncation=True,
                    max_length=datamodule.max_seq_length,
                    padding="max_length",
                    return_offsets_mapping=True,
                    return_tensors="pt")
            entity_token_idxs = datamodule.get_token_indices_from_char_span(
                    example["entity_char_span"], encoded["offset_mapping"][0])
            input_ids = encoded["input_ids"][0]
            reconstructed_entity = datamodule.tokenizer.decode(
                    input_ids[entity_token_idxs])
            # rm all whitespace b/c decode() doesn't always
            #  reconstruct it correctly
            reconstructed_entity = re.sub(r'\s+', '', reconstructed_entity)
            start_char, end_char = example["entity_char_span"]
            orig_entity = example["text"][start_char:end_char]
            orig_entity = re.sub(r'\s+', '', orig_entity)
            msg = f"{reconstructed_entity.lower()} != {orig_entity.lower()}"
            assert reconstructed_entity.lower() == orig_entity.lower(), msg
    del datamodule


@test_logger
def test_mask_hidden(config):
    data_kwargs = {
        "entity_markers": None,
        "batch_size": 1,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore changing type of config parameters
        datamodule = load_datamodule_from_config(config, **data_kwargs)
        # Ignore "No test set found"
        datamodule.setup()
    msg = "datamodule.entity_markers expected to be None"
    assert datamodule.entity_markers is None, msg

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "first-last",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    msg = "model.use_entity_spans is not True"
    assert model.use_entity_spans is True, msg
    msg = "entity_pool_fn != 'first-last'"
    assert model.entity_pool_fn == "first-last", msg

    insize = outsize = model.bert_config.hidden_size
    pooler = TokenEmbeddingPooler(insize, outsize, model.entity_pool_fn)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            this_batch_size = batch["encodings"]["input_ids"].size(0)
            h = torch.randn(this_batch_size, config.max_seq_length,
                            model.bert_config.hidden_size)
            input_ids = batch["encodings"]["input_ids"][0]
            entity_ids = input_ids[batch["entity_token_idxs"][0]]
            token_mask = pooler.get_token_mask_from_indices(
                    batch["entity_token_idxs"], h.size())
            masked_ids = input_ids[token_mask[0, :, 0].bool()]
            msg = "not all(masked_ids == entity_ids)"
            assert (masked_ids == entity_ids).all(), msg
    del datamodule
    del model
    del pooler


@test_logger
def test_mask_hidden_marked(config):
    data_kwargs = {
        "entity_markers": '@',
        "batch_size": 1,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore changing type of config parameters
        datamodule = load_datamodule_from_config(config, **data_kwargs)
        # Ignore "No test set found"
        datamodule.setup()
    msg = "datamodule.entity_markers expected to be ('@', '@')"
    assert datamodule.entity_markers == ('@', '@'), msg

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "first-last",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    msg = "model.use_entity_spans is not True"
    assert model.use_entity_spans is True, msg
    msg = "entity_pool_fn != 'first-last'"
    assert model.entity_pool_fn == "first-last", msg

    start_marker = datamodule.train.entity_markers[0]
    end_marker = datamodule.train.entity_markers[1]
    start_id = datamodule.tokenizer.convert_tokens_to_ids([start_marker])[0]
    end_id = datamodule.tokenizer.convert_tokens_to_ids([end_marker])[0]
    insize = outsize = model.bert_config.hidden_size
    pooler = TokenEmbeddingPooler(insize, outsize, model.entity_pool_fn)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            this_batch_size = batch["encodings"]["input_ids"].size(0)
            h = torch.randn(this_batch_size, config.max_seq_length,
                            model.bert_config.hidden_size)
            token_mask = pooler.get_token_mask_from_indices(
                    batch["entity_token_idxs"], h.size())
            input_ids = batch["encodings"]["input_ids"][0]
            masked_ids = input_ids[token_mask[0, :, 0].bool()]
            msg = "masked_ids != (start_id, end_id)"
            assert masked_ids[0] == start_id and masked_ids[-1] == end_id, msg
    del datamodule
    del model
    del pooler


@test_logger
def test_pool_entity_embeddings_max(config):
    data_kwargs = {
        "entity_markers": None,
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    msg = "entity_markers is not None"
    assert datamodule.entity_markers is None, msg

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "max",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    msg = "use_entity_spans is not True"
    assert model.use_entity_spans is True, msg
    msg = "entity_pool_fn != 'max'"
    assert model.entity_pool_fn == "max", msg

    insize = outsize = model.bert_config.hidden_size
    pooler = TokenEmbeddingPooler(insize, outsize, model.entity_pool_fn)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            this_batch_size = batch["encodings"]["input_ids"].size(0)
            h = torch.randn(this_batch_size, config.max_seq_length,
                            model.bert_config.hidden_size)
            pooled = pooler(h, batch["entity_token_idxs"])
            msg = "some pooled == 0"
            assert (pooled == torch.tensor(0.)).any() == torch.tensor(False), msg  # noqa
    del datamodule
    del model
    del pooler


@test_logger
def test_pool_entity_embeddings_mean(config):
    data_kwargs = {
        "entity_markers": None,
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    msg = "entity_markers is not None"
    assert datamodule.entity_markers is None, msg

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "mean",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    msg = "use_entity_spans is not True"
    assert model.use_entity_spans is True
    msg = "entity_pool_fn != 'mean'"
    assert model.entity_pool_fn == "mean"

    insize = outsize = model.bert_config.hidden_size
    pooler = TokenEmbeddingPooler(insize, outsize, model.entity_pool_fn)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            this_batch_size = batch["encodings"]["input_ids"].size(0)
            h = torch.randn(this_batch_size, config.max_seq_length,
                            model.bert_config.hidden_size)
            pooled = pooler(h, batch["entity_token_idxs"])
            msg = "pooled contains nan"
            assert not torch.isnan(pooled).all(), msg
            msg = "pooled contains inf"
            assert not torch.isinf(pooled).all(), msg
    del datamodule
    del model
    del pooler


@test_logger
def test_pool_entity_embeddings_first(config):
    data_kwargs = {
        "entity_markers": None,
        "batch_size": 2,
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    msg = "entity_markers is not None"
    assert datamodule.entity_markers is None, msg

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "first",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    msg = "use_entity_spans is not True"
    assert model.use_entity_spans is True, msg
    msg = "entity_pool_fn != 'first'"
    assert model.entity_pool_fn == "first", msg

    insize = outsize = model.bert_config.hidden_size
    pooler = TokenEmbeddingPooler(insize, outsize, model.entity_pool_fn)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            this_batch_size = batch["encodings"]["input_ids"].size(0)
            h = torch.randn(this_batch_size, config.max_seq_length,
                            model.bert_config.hidden_size)
            pooled = pooler(h, batch["entity_token_idxs"])
            msg = "pooled contains nan"
            assert not torch.isnan(pooled).all(), msg
            msg = "pooled contains inf"
            assert not torch.isinf(pooled).all(), msg
    del datamodule
    del model
    del pooler


@test_logger
def test_pool_entity_embeddings_last(config):
    data_kwargs = {
        "entity_markers": None,
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    msg = "entity_markers is not None"
    assert datamodule.entity_markers is None, msg

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "last",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    msg = "use_entity_spans is not True"
    assert model.use_entity_spans is True, msg
    msg = "entity_pool_fn != 'last'"
    assert model.entity_pool_fn == "last", msg

    insize = outsize = model.bert_config.hidden_size
    pooler = TokenEmbeddingPooler(insize, outsize, model.entity_pool_fn)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            this_batch_size = batch["encodings"]["input_ids"].size(0)
            h = torch.randn(this_batch_size, config.max_seq_length,
                            model.bert_config.hidden_size)
            pooled = pooler(h, batch["entity_token_idxs"])
            msg = "pooled contains nan"
            assert not torch.isnan(pooled).all(), msg
            msg = "pooled contains inf"
            assert not torch.isinf(pooled).all(), msg
    del datamodule
    del model
    del pooler


@test_logger
def test_pool_entity_embeddings_first_last(config):
    data_kwargs = {
        "entity_markers": None,
    }
    datamodule = load_datamodule_from_config(config, **data_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore "No test set found"
        datamodule.setup()
    msg = "entity_markers is not None"
    assert datamodule.entity_markers is None, msg

    model_kwargs = {
        "use_entity_spans": True,
        "entity_pool_fn": "first-last",
    }
    model = BertMultiHeadedSequenceClassifier.from_config(
        config, datamodule, **model_kwargs)
    msg = "use_entity_spans is not True"
    assert model.use_entity_spans is True, msg
    msg = "entity_pool_fn != 'first-last'"
    assert model.entity_pool_fn == "first-last", msg

    insize = outsize = model.bert_config.hidden_size
    pooler = TokenEmbeddingPooler(insize, outsize, model.entity_pool_fn)
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            this_batch_size = batch["encodings"]["input_ids"].size(0)
            h = torch.randn(this_batch_size, config.max_seq_length,
                            model.bert_config.hidden_size)
            pooled = pooler(h, batch["entity_token_idxs"])
            msg = "pooled contains nan"
            assert not torch.isnan(pooled).all(), msg
            msg = "pooled contains inf"
            assert not torch.isinf(pooled).all(), msg
    del datamodule
    del model
    del pooler


@test_logger
def test_scheduled_dataset_sampler(config, project_home):
    data_kwargs = {
        "batch_size": 16,
        "max_train_examples": None,
        "auxiliary_data": {
            "n2c2Assertion": {
                "dataset_name": "n2c2Assertion",
                "data_dir": f"{project_home}/auxiliary_data/n2c2_2010_concept_assertion_relation/combined/ast_brat/",  # noqa
                "sentences_dir": f"{project_home}/auxiliary_data/n2c2_2010_concept_assertion_relation/combined/segmented/",  # noqa
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
    msg = "datamodule is not CombinedDataModule"
    assert isinstance(datamodule, combined.CombinedDataModule), msg

    sampler = datamodule.train_dataloader().batch_sampler
    msg = "sampler is not ScheduledWeightedSampler"
    assert isinstance(sampler, combined.ScheduledWeightedSampler), msg

    ntrials = 10
    sampler_lens = np.zeros(sampler.max_steps)
    batch_lens = np.zeros((ntrials, sampler.max_steps))
    for i in range(ntrials):
        sampler.reset()
        for j in range(sampler.max_steps):
            for (k, batch) in enumerate(sampler):
                # Could be less than if we exhaust a dataset this step
                msg = "exceeded batch size"
                assert len(batch) <= sampler.batch_size, msg
                assert len(batch) <= datamodule.batch_size, msg
                if k == 0:
                    sampler_lens[j] = sampler.target_samples_this_epoch
                batch_lens[i, j] += 1

    batch_matches_sampler = (batch_lens == sampler_lens).sum(axis=0)
    msg = "batch_lens != sampler_lens"
    assert (batch_matches_sampler == ntrials).all(), msg

    sampler.reset()
    dataset_lens = [len(ds) for ds in sampler.dataset.datasets]
    gold_lens = np.array(
        [sampler._estimate_length(dataset_lens, sampler.get_dataset_probs(s))
         for s in range(sampler.max_steps)]
    )
    msg = "found incorrect lengths"
    assert (gold_lens == sampler_lens).all(), msg


@test_logger
def test_stickland_murray_dataset_sampler(config, project_home):
    data_kwargs = {
        "batch_size": 16,
        "max_train_examples": None,
        "auxiliary_data": {
            "n2c2Assertion": {
                "dataset_name": "n2c2Assertion",
                "data_dir": f"{project_home}/auxiliary_data/n2c2_2010_concept_assertion_relation/combined/ast_brat/",  # noqa
                "sentences_dir": f"{project_home}/auxiliary_data/n2c2_2010_concept_assertion_relation/combined/segmented/",  # noqa
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
    msg = "datamodule is not CombinedDataModule"
    assert isinstance(datamodule, combined.CombinedDataModule), msg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler = datamodule.train_dataloader().batch_sampler
    msg = "sampler is not SticklandMurraySampler"
    assert isinstance(sampler, combined.SticklandMurraySampler), msg

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
                msg = "exceeded batch size"
                assert len(batch) <= sampler.batch_size, msg
                assert len(batch) <= datamodule.batch_size, msg
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
        msg = "probabilities do not decrease"
        assert start_diff > end_diff, msg

    batch_matches_sampler = (batch_lens == sampler_lens).sum(axis=0)
    msg = "batch_lens != sampler_lens"
    assert (batch_matches_sampler == ntrials).all(), msg

    sampler.reset()
    dataset_lens = [len(ds) for ds in sampler.dataset.datasets]
    gold_lens = np.array(
        [sampler._estimate_length(dataset_lens, sampler.get_dataset_probs(s))
         for s in range(sampler.max_steps)]
    )
    msg = "found incorrect lengths"
    assert (gold_lens == sampler_lens).all(), msg


@test_logger
def test_levitate_encodings(config, i=0):
    data_kwargs = {
        "entity_markers": ["[unused0]", "[unused1]"],
        "use_levitated_markers": True,
        "batch_size": 1,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        datamodule = load_datamodule_from_config(config, **data_kwargs)
        # Ignore "No test set found"
        datamodule.setup()
    msg = "entity_markers != ['[unused0]', '[unused1]']"
    assert datamodule.entity_markers == ["[unused0]", "[unused1]"], msg
    msg = "use_levitated_markers is not True"
    assert datamodule.use_levitated_markers is True, msg
    for dl in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        for batch in dl:
            msg = "batch doesn't contain levitated markers"
            assert "levitated_marker_idxs" in batch.keys(), msg
            marker_idxs = set(batch["levitated_marker_idxs"][0])
            entity_idxs = set(batch["entity_token_idxs"][0])
            msg = "levitated markers overlap with entity"
            assert len(marker_idxs.intersection(entity_idxs)) == 0, msg
            # See src/data/base.BasicBertDataModule.levitate_encodings()
            #  for an explanation of this computation.
            max_num_markers = datamodule.levitate_window_size * 4
            expected_seq_length = datamodule.max_seq_length + max_num_markers
            for (key, val) in batch["encodings"].items():
                # [batch_size, seq_length, [hidden_dim]]
                msg = f"Value of {key} is the wrong size! {expected_seq_length} vs. {val.size(1)}"  # noqa
                assert val.size(1) == expected_seq_length, msg


if __name__ == "__main__":
    args = parse_args()
    run(args.project_home)
