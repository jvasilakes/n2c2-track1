# BERT and BERT-derivative baseline models

## Setup

```
conda env create -f environment.yaml
conda activate n2c2
```

You'll also need to install `probabll`, which is included as a submodule.

```
cd models/common/dists.pt
python setup.py develop
```

## Usage

Generate a default experiment config file with
```
python config.py new path/to/new.yaml
```

Edit this file to your liking. Some less obvious options are explained below.
Then run

```
python run.py train /path/to/config.yaml
python run.py validate /path/to/logs/experiment_name/version/config.yaml
```

### Rerunning/recreating an experiment

The config files used for all experiments are logged. If you're on CSF3, you can see the location of the current best models by 

```
ls -l configs/current_best
```

You can rerun one of the experiments by simply calling `python run.py train /path/to/config.yaml`, where the config.yaml is the linked file from the above `ls -l` command.


### Updating config files

It can happen that config files go out of date and are no longer compatible with the current scripts. You can update a yaml file, keeping the same parameters but using the most up-to-date format with

```
python config.py update -f /path/to/config.yaml
```

This command will save the original config file at `/path/to/config.yaml.orig`, in case you still need it.

P.S., you can also use `config.py update` to change one or more parameters of a batch of config files. E.g., 

```
python config.py update -k window_size 2 -k batch_size 16 -f /path/to/many/*.yaml
```
will update all files matched by `/path/to/many/*yaml` to have `window_size: 2` and `batch_size: 16`.

Each `-k` specifies a parameter to update and expects a space-separated key-value pair.
All arguments after `-f` are files to update with the specified `-k`s.  As shown, you can use wildcards/globs to pass a bunch of files.

If `config.py update` finds something incompatible in the given config files, it will "fix" it by either ignoring it (if its an unknown key) or setting it to its default value (if its an unsupported/incorrectly typed value). In both cases it will give you a clear warning, in case you need to modify things manually.


## Config Parameters

Below are descriptions for some of the less obvious parameters you can specify in the config files.

### Experiment options
* `name`: The name of this experiment
* `description`: A brief description of what sets this experiment apart. E.g., model structure or hyperparamters.
* `logdir`: The base log directory for this experiment. The actual experiment will be logged to `logdir/name/version_[0-9]+?`, where the version number is determined by Pytorch Lightning.

### Data options
* `dataset_name`: `str` corresponding to a Dataset class in `data`. To get available n2c2 dataset names, run `python -m data.n2c2`
* `data_dir`: `/path/to/n2c2Track1TrainingData/data`
* `sentences_dir`: `/path/to/n2c2Track1TrainingData/segmented` Assumes JSON lines format as output by `/path/to/n2c2Track1TrainingData/segmented/scripts/run_biomedicus_sentences.py`.
* `tasks_to_load`: List of the context classification tasks to perform. Valid options are `"all"`, or any subset of `["Action", "Actor", "Certainity", "Negation", "Temporality"]`
* `window_size`: Number of additional sentences before and after the target sentence to use. Default 0.
* `max_train_examples`: `null` or `int`. Limit the number of training examples. Useful for debugging.
* `auxiliary_data`: This is a dict keyed by datamodule names in `data.DATAMODULE_LOOKUP` with values exactly the same as the Data Options above. If `auxiliary_data` is not empty, the primary dataset and all auxiliary datasets are loaded into a `data.combined.CombinedDataModule`. Each batch sampled from a `CombinedDataModule` contains examples from a single dataset. Batches/datasets are sampled according to `dataset_sample_strategy`, described below.
* `dataset_sample_strategy`: Only used if `auxiliary_data` is not empty. Possible values are `concat`, `fixed:{int}`, `proportional`, and `annealed`.
  - `concat`: Simply concatenate all the datasets and run through them all during training. Shuffles both the example indices within each dataset and the order in which the datasets are sampled.
  - `fixed:{int}`: Where `{int}` is an integer within 0 to 100 that represents the probability of sampling one of the auxiliary datasets. This probability is applied *per auxiliary dataset*. So, for example, specifying `fixed:10` with two auxiliary datasets means that on average 20% of the training steps per epoch will come from either of the auxiliary datasets.
  - `proportional`: Sample from each dataset (main + all auxiliaries) with probabilities proportional to their number of training examples.
  - `annealed`: Like proportional, except the probabilities are annealed per epoch to become uniform after 10 epochs. Introduced by [Stickland and Murray, 2019][1].

### Model options
* `model_name`: Model to run, corresponding to entries in `models.MODEL_LOOKUP.keys()`.
* `bert_model_name_or_path`: BERT or BERT-derivative to load. E.g. `bert-base-uncased`.
* `freeze_pretrained`: If `true`, freeze the BERT model weights.
* `use_entity_spans`: If `true`, use the pooled embeddings from the target entity span for classification, rather than the full input.
* `entity_pool_fn`: `"max"`, `"mean"`, or `"first"`, `"last"`, `"first-last"`. If `use_entity_spans is True` how to pool the entity token embeddings before sending them to the classification head(s). `"max"`: take the maximum of each embedding dimension over the entity tokens; `"mean"`: take the mean of each embedding dimension over the entity tokens; `"first"`: take the embedding of the first entity token; `"last"`: take the embedding of the last entity token; `"first-last"`: take the embedding of the first and last entity tokens.
* `mark_entities`: If `true`, surround the entities with `@` symbols. Note that these are considered entity tokens by `entity_pool_fn`. So, `entity_pool_fn: first-last` will take the embeddings of the `@` symbols.

### Loss options
* `classifier_loss_fn`: Name of the loss function to use for the classifiers. See the current supported loss functions by running `python models/losses.py`
* `classifier_loss_kwargs`: Dictionary of keyword arguments to pass to the classifier loss function. Generally, these are the same as the arguments to the class instance given by `python models/losses.py`, but there are the following exceptions:
  - `class_weights: "balanced"`: Computes balanced class weights for the loaded tasks and passes them to the `weight` argument of the loss class (used with `torch.nn.CrossEntropyLoss`)
* `mask_loss_fn`: Name of the loss function to use for the stochastic masks. Only used with `model_name: "bert-rationale-classifier`.
* `mask_loss_kwargs`: Analogous to `classifier_loss_kwargs` for the stochastic mask loss function.

### Training options
* `sample_strategy`: `null` or `"weighted"`. How to sample training examples. If `null`, use shuffled batch sampling. If `"weighted"`, sample according to inverse probability of the examples' label(s) in the train set using `torch.utils.data.WeightedRandomSampler`.


# Results

See [Results.md](https://github.com/jvasilakes/n2c2-track1/blob/master/context/bert_baselines/Results.md).


[1]: https://proceedings.mlr.press/v97/stickland19a.html
