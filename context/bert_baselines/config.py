import os
import yaml
import argparse
import warnings
from typing import List, Union
from collections import OrderedDict


# Be able to yaml.dump an OrderedDict
# https://gist.github.com/oglops/c70fb69eef42d40bed06
def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


yaml.Dumper.add_representer(OrderedDict, dict_representer)


class ConfigError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
            max_train_examples: int = -1,
            window_size: int = 0,
            max_seq_length: int = 128,
            dropout_prob: float = 0.1,
            batch_size: int = 1,
            sample_strategy: str = 'none',
            use_entity_spans: bool = False,
            entity_pool_fn: str = "max",
            max_epochs: int = 1,
            class_weights: str = 'none',
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            gradient_clip_val: float = 10.0,
            freeze_pretrained: bool = False,
            random_seed: int = 0,
            run_validate: bool = True,
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
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.sample_strategy = sample_strategy
        self.use_entity_spans = use_entity_spans
        self.entity_pool_fn = entity_pool_fn
        self.max_epochs = max_epochs
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.freeze_pretrained = freeze_pretrained
        self.random_seed = random_seed

        self.organize()
        if run_validate is True:
            self.validate()

    def __str__(self):
        organized_params_with_values = OrderedDict()
        for (param_type, param_names) in self._organized_params.items():
            organized_params_with_values[param_type] = OrderedDict()
            for name in param_names:
                organized_params_with_values[param_type][name] = getattr(self, name)  # noqa
        yaml_str = yaml.dump(organized_params_with_values, Dumper=yaml.Dumper,
                             default_flow_style=False)
        yaml_str = '  ' + yaml_str.replace('\n', '\n  ')
        return "ExperimentConfig\n----------------\n" + yaml_str

    def _validate_param(self, param_name, valid_values,
                        default_value, errors="raise"):
        param_value = getattr(self, param_name)
        if param_value not in valid_values:
            if errors == "raise":
                raise ConfigError(
                    f"Unsupported {param_name} '{param_value}'. Expected one of {valid_values}.")  # noqa
            elif errors == "fix":
                setattr(param_name, default_value)
                warnings.warn(f"{param_name} set to default `{default_value}`")
            else:
                raise ValueError(f"Unknown errors value {errors}. Expected 'fix' or 'raise'.")  # noqa

    def validate(self, errors="raise"):
        valid_entity_pool_fns = ["mean", "max", "first"]
        self._validate_param("entity_pool_fn", valid_entity_pool_fns,
                             default_value="mean", errors=errors)

        valid_sample_strategies = ["none", "weighted"]
        self._validate_param("sample_strategy", valid_sample_strategies,
                             default_value="none", errors=errors)

        valid_class_weights = ["none", "balanced"]
        self._validate_param("class_weights", valid_class_weights,
                             default_value="none", errors=errors)

        used_params = set([key for key in self.__dict__.keys()
                           if not key.startswith('_')])
        organized_params = set(
            [key for param_set in self._organized_params.values()
             for key in param_set])
        used_but_not_organized = used_params.difference(organized_params)
        organized_but_not_used = organized_params.difference(used_params)
        if len(used_but_not_organized) > 0:
            msg = f"""The following parameters were defined but not added to
            ExperimentConfig.organize(): {used_but_not_organized}"""
            raise ConfigError(msg)
        if len(organized_but_not_used) > 0:
            msg = f"""The following parameters were added to
            ExperimentConfig.organize() but not defined:
            {organized_but_not_used}"""
            raise ConfigError(msg)

    def organize(self):
        self._organized_params = OrderedDict({
            "Experiment": [
                "name",
                "description",
                "logdir",
                "random_seed",
            ],
            "Data": [
                "data_dir",
                "sentences_dir",
                "tasks_to_load",
                "window_size",
                "max_train_examples",
            ],
            "Model": [
                "model_name",
                "bert_model_name_or_path",
                "freeze_pretrained",
                "use_entity_spans",
                "entity_pool_fn",
                "max_seq_length",
                "dropout_prob",
            ],
            "Training": [
                "batch_size",
                "sample_strategy",
                "lr",
                "weight_decay",
                "gradient_clip_val",
                "max_epochs",
                "class_weights",
            ]
        })

    def update(self, key, value):
        if key == "tasks_to_load":
            raise NotImplementedError("updating tasks_to_load not yet implemented.")  # noqa
        param_type = type(getattr(self, key))
        try:
            cast_value = param_type(value)
        except ValueError:
            msg = f"Unable to cast value '{value}' to expected type {param_type} for {key}"  # noqa
            raise ConfigError(msg)
        setattr(self, key, cast_value)
        self.validate()

    @classmethod
    def from_yaml_file(cls, filepath, errors="raise"):
        with open(filepath, 'r') as inF:
            config = yaml.safe_load(inF)
            conf = cls(**config, run_validate=False)
            conf.validate(errors=errors)
            return conf

    @classmethod
    def write_default_config(cls, outpath):
        cls().save_to_yaml(outpath)

    def save_to_yaml(self, outpath):
        with open(outpath, 'w') as outF:
            for (param_type, param_names) in self._organized_params.items():
                outF.write(f"# {param_type}\n")
                params_dict = {name: self.__dict__[name]
                               for name in param_names}
                yaml.dump(params_dict, outF)
                outF.write('\n')


def update_config_file(filepath, **update_kwargs):
    conf = ExperimentConfig.from_yaml_file(filepath, errors="fix")
    for (key, value) in update_kwargs.items():
        conf.update(key, value)
    os.rename(filepath, f"{filepath}.orig")
    conf.save_to_yaml(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    newconf_parser = subparsers.add_parser(
        "new", help="Save a default config file to the given filepath")
    newconf_parser.add_argument("filepath", type=str,
                                help="Where to save the config file.")

    update_parser = subparsers.add_parser(
        "update",
        help="""Update one or more config files
                with the given --key=value pairs""")
    update_parser.add_argument("-k", "--kwarg", nargs=2, action="append",
                               help="""E.g., -k max_seq_length 128""")
    update_parser.add_argument("-f", "--files", nargs='+', type=str,
                               help="""Files to update""")

    args = parser.parse_args()
    if args.command == "new":
        ExperimentConfig.write_default_config(args.filepath)
    elif args.command == "update":
        if args.kwarg is None:
            update_kwargs = {}
        else:
            update_kwargs = dict(args.kwarg)
        for filepath in args.files:
            update_config_file(filepath, **update_kwargs)
