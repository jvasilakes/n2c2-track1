# BERT and BERT-derivative baseline models

## Usage

Generate a default experiment config file.
```
python config.py --filepath configs/new.yaml
```
A nicely formatted example is at `configs/test.yaml`.

Edit this file to your liking. Some less obvious options are explained below.
Then run

```
python3 run.py train /path/to/config.yaml
python3 run.py validate /path/to/logs/experiment_name/version/config.yaml
```


### Experiment options
* `name`: The name of this experiment
* `description`: A brief description of what sets this experiment apart. E.g., model structure or hyperparamters.

### Data options
* `data_dir`: `/path/to/n2c2Track1TrainingData/data`
* `sentences_dir`: `/path/to/n2c2Track1TrainingData/segmented` Assumes JSON lines format as output by `/path/to/n2c2Track1TrainingData/segmented/scripts/run_biomedicus_sentences.py`.
* `tasks_to_load`: List of the context classification tasks to perform. Valid options are `"all"`, or any subset of `["Action", "Actor", "Certainity", "Negation", "Temporality"]`
* `max_train_examples`: `null` or `int`. Limit the number of training examples. Useful for debugging.

### Model options
* `model_name`: Model to run, corresponding to entries in `model.MODEL_LOOKUP.keys()`.
* `bert_model_name_or_path`: BERT or BERT-derivative to load. E.g. `bert-base-uncased`.
* `use_entity_spans`: If `true`, use the pooled embeddings from the target entity span for classification, rather than the full input.
* `entity_pool_fn`: `"max"` or `"mean"`. If `use_entity_spans is True` how to pool the entity token embeddings before sending them to the classification head(s).


# Results
## bert-base-uncased
### Pooled output
#### Individual tasks
| Action  | P     | R     | F1    | Actor   | P     | R     | F1    | Cert    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.715 | 0.715 | 0.715 | micro   | 0.869 | 0.869 | 0.869 | micro   | 0.851 | 0.851 | 0.851 |
| macro   | 0.647 | 0.541 | 0.573 | macro   | 0.438 | 0.473 | 0.455 | macro   | 0.708 | 0.671 | 0.688 |

| Neg     | P     | R     | F1    | Temp    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.973 | 0.973 | 0.973 | micro   | 0.778 | 0.778 | 0.778 |
| macro   | 0.618 | 0.618 | 0.618 | macro   | 0.806 | 0.637 | 0.673 |


#### All tasks
| Action  | P     | R     | F1    | Actor   | P     | R     | F1    | Cert    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.113 | 0.113 | 0.113 | micro   | 0.317 | 0.317 | 0.317 | micro   | 0.471 | 0.471 | 0.471 |
| macro   | 0.112 | 0.144 | 0.086 | macro   | 0.393 | 0.501 | 0.251 | macro   | 0.323 | 0.333 | 0.267 |

| Neg     | P     | R     | F1    | Temp    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.647 | 0.647 | 0.647 | micro   | 0.778 | 0.778 | 0.778 |
| macro   | 0.486 | 0.329 | 0.393 | macro   | 0.786 | 0.629 | 0.663 |


## Bio-ClincalBERT
### Pooled Output
#### Individual tasks
| Action  | P     | R     | F1    | Actor   | P     | R     | F1    | Cert    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.710 | 0.710 | 0.710 | micro   | 0.887 | 0.887 | 0.887 | micro   | 0.896 | 0.896 | 0.896 |
| macro   | 0.725 | 0.617 | 0.620 | macro   | 0.468 | 0.426 | 0.438 | macro   | 0.829 | 0.736 | 0.775 |

| Neg     | P     | R     | F1    | Temp    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.946 | 0.946 | 0.946 | micro   | 0.778 | 0.778 | 0.778 |
| macro   | 0.543 | 0.604 | 0.557 | macro   | 0.780 | 0.629 | 0.661 |

#### All Tasks
| Action  | P     | R     | F1    | Actor   | P     | R     | F1    | Cert    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.113 | 0.113 | 0.113 | micro   | 0.176 | 0.176 | 0.176 | micro   | 0.308 | 0.308 | 0.308 |
| macro   | 0.245 | 0.167 | 0.099 | macro   | 0.351 | 0.383 | 0.120 | macro   | 0.158 | 0.110 | 0.123 |

| Neg     | P     | R     | F1    | Temp    | P     | R     | F1    |
|---------|-------|-------|-------|---------|-------|-------|-------|
| micro   | 0.181 | 0.181 | 0.181 | micro   | 0.769 | 0.769 | 0.769 |
| macro   | 0.481 | 0.338 | 0.159 | macro   | 0.556 | 0.532 | 0.533 |
