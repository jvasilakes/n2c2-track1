# BERT and BERT-derivative baseline models

## Usage

Generate a default experiment config file.
```
python config.py --filepath configs/new.yaml
```
A nicely formatted example is at `configs/test.yaml`.

Edit this file to your liking. Some less obvious options are explained below.

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