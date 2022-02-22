import yaml
import argparse
from typing import List, Union


class ExperimentConfig(object):

    def __init__(
            self,
            name: str = "default_experiment",
            description: str = '',
            logdir: str = "logs/",
            data_dir: str = '',
            sentences_dir: str = '',
            model_name: str = '',
            bert_model_name_or_path: str = '',
            tasks_to_load: Union[List[str], str] = "all",
            max_train_examples: int = None,
            window_size: int = 0,
            max_seq_length: int = 128,
            batch_size: int = 1,
            sample_strategy: str = None,
            use_entity_spans: bool = False,
            entity_pool_fn: str = "max",
            max_epochs: int = 1,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            gradient_clip_val: float = 10.0,
            freeze_pretrained: bool = False,
            random_seed: int = 0,
            ):
        self.name = name
        self.description = description
        self.logdir = logdir
        self.data_dir = data_dir
        self.sentences_dir = sentences_dir
        self.model_name = model_name
        self.bert_model_name_or_path = bert_model_name_or_path
        self.tasks_to_load = tasks_to_load
        self.max_train_examples = max_train_examples
        self.window_size = window_size
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.sample_strategy = sample_strategy
        self.use_entity_spans = use_entity_spans
        self.entity_pool_fn = entity_pool_fn
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.freeze_pretrained = freeze_pretrained
        self.random_seed = random_seed

        self.validate()

    def __str__(self):
        yaml_str = yaml.dump(self.__dict__)
        yaml_str = '  ' + yaml_str.replace('\n', '\n  ')
        return "ExperimentConfig\n" + yaml_str

    def validate(self):
        assert self.entity_pool_fn in ["mean", "max"]
        assert self.sample_strategy in [None, "weighted"]

    @classmethod
    def from_yaml_file(cls, filepath):
        with open(filepath, 'r') as inF:
            config = yaml.safe_load(inF)
            return cls(**config)

    @classmethod
    def write_default_config(cls, outpath):
        cls().save_to_yaml(outpath)

    def save_to_yaml(self, outpath):
        with open(outpath, 'w') as outF:
            yaml.dump(self.__dict__, outF)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default=None,
                        help="Write a default config file to filepath")
    args = parser.parse_args()
    if args.filepath is not None:
        ExperimentConfig.write_default_config(args.filepath)
