import os
import yaml
import argparse
import warnings
from typing import List, Dict, Union
from collections import OrderedDict, defaultdict


# Be able to yaml.dump an OrderedDict
# https://gist.github.com/oglops/c70fb69eef42d40bed06
def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


yaml.Dumper.add_representer(OrderedDict, dict_representer)


class ConfigError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExperimentConfig(object):

    @staticmethod
    def organized_param_names():
        """
        Any arguments to __init__ must be included here.
        """
        return OrderedDict({
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
                "auxiliary_data",
                "dataset_sample_strategy",
            ],
            "Model": [
                "model_name",
                "bert_model_name_or_path",
                "freeze_pretrained",
                "use_entity_spans",
                "entity_pool_fn",
                "mark_entities",
                "max_seq_length",
                "dropout_prob",
            ],
            "Losses": [
                "classifier_loss_fn",
                "classifier_loss_kwargs",
                "mask_loss_fn",
                "mask_loss_kwargs",
            ],
            "Training": [
                "batch_size",
                "sample_strategy",
                "lr",
                "weight_decay",
                "gradient_clip_val",
                "max_epochs",
            ]
        })

    @classmethod
    def param_names(cls):
        return [name for names in cls.organized_param_names().values()
                for name in names]

    def __init__(
            self,
            # Experiment
            name: str = "default_experiment",
            description: str = '',
            logdir: str = "logs/",
            random_seed: int = 0,
            # Data
            # The kwargs below always specify the main n2c2 dataset
            data_dir: str = '',
            sentences_dir: str = '',
            tasks_to_load: Union[List[str], str] = "all",
            window_size: int = 0,
            max_train_examples: int = -1,
            # {dataset_name: dict()} where dict has name args as above.
            # dataset_name is used to determine which datamodule to use.
            auxiliary_data: Dict[str, List] = None,
            # Ignored if auxiliary_data is None
            dataset_sample_strategy: str = "proportional",
            # Model
            model_name: str = '',
            bert_model_name_or_path: str = '',
            freeze_pretrained: bool = False,
            use_entity_spans: bool = False,
            entity_pool_fn: str = "max",
            mark_entities: bool = False,
            max_seq_length: int = 128,
            dropout_prob: float = 0.1,
            # Losses
            classifier_loss_fn: str = "cross-entropy",
            classifier_loss_kwargs: dict = None,
            mask_loss_fn: str = "ratio",
            mask_loss_kwargs: dict = None,
            # Training
            batch_size: int = 1,
            sample_strategy: str = None,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            gradient_clip_val: float = 10.0,
            max_epochs: int = 1,
            # Validate params
            run_validate: bool = True,
            ):
        # Experiment
        self.name = name
        self.description = description
        self.logdir = logdir
        self.random_seed = random_seed
        # Data
        self.data_dir = data_dir
        self.sentences_dir = sentences_dir
        self.tasks_to_load = tasks_to_load
        self.window_size = window_size
        self.max_train_examples = max_train_examples
        self.auxiliary_data = auxiliary_data or {}
        self.dataset_sample_strategy = dataset_sample_strategy
        # Model
        self.model_name = model_name
        self.bert_model_name_or_path = bert_model_name_or_path
        self.freeze_pretrained = freeze_pretrained
        self.use_entity_spans = use_entity_spans
        self.entity_pool_fn = entity_pool_fn
        self.mark_entities = mark_entities
        self.max_seq_length = max_seq_length
        self.dropout_prob = dropout_prob
        # Losses
        self.classifier_loss_fn = classifier_loss_fn
        self.classifier_loss_kwargs = classifier_loss_kwargs or {}
        self.mask_loss_fn = mask_loss_fn
        self.mask_loss_kwargs = mask_loss_kwargs or {}
        # Training
        self.batch_size = batch_size
        self.sample_strategy = sample_strategy
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.max_epochs = max_epochs

        if run_validate is True:
            self.validate()

    def __str__(self):
        organized_params_with_values = OrderedDict()
        for (param_type, param_names) in self.organized_param_names().items():
            organized_params_with_values[param_type] = OrderedDict()
            for name in param_names:
                val = getattr(self, name)
                # Don't want to print out all the weird yaml
                # encodings for python objects.
                if isinstance(val, (OrderedDict, defaultdict)):
                    val = dict(val)
                organized_params_with_values[param_type][name] = val
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
                setattr(self, param_name, default_value)
                warnings.warn(f"{param_name} set to default `{default_value}` from unsupported value `{param_value}`.")  # noqa
            else:
                raise ValueError(f"Unknown errors value {errors}. Expected 'fix' or 'raise'.")  # noqa

    def validate(self, errors="raise"):
        valid_entity_pool_fns = ["mean", "max", "first", "last", "first-last"]
        self._validate_param("entity_pool_fn", valid_entity_pool_fns,
                             default_value="mean", errors=errors)

        valid_sample_strategies = [None, "weighted"]
        self._validate_param("sample_strategy", valid_sample_strategies,
                             default_value=None, errors=errors)

        used_params = set([key for key in self.__dict__.keys()
                           if not key.startswith('_')])
        valid_params = set(self.param_names())
        used_but_not_organized = used_params.difference(valid_params)
        organized_but_not_used = valid_params.difference(used_params)
        if len(used_but_not_organized) > 0:
            msg = f"""The following parameters were defined but not added to
            ExperimentConfig.organized_param_names: {used_but_not_organized}"""
            raise ConfigError(msg)
        if len(organized_but_not_used) > 0:
            msg = f"""The following parameters were added to
            ExperimentConfig.organized_param_names but not defined:
            {organized_but_not_used}"""
            raise ConfigError(msg)

    def update(self, key, value):
        if key in ["tasks_to_load", "classifier_loss_kwargs", "mask_loss_kwargs"]:  # noqa
            raise NotImplementedError(f"updating '{key}' not yet implemented.")
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
            to_del = set()
            for key in config.keys():
                if key not in cls.param_names():
                    if errors == "raise":
                        raise ValueError(f"Unsupported config parameter '{key}'")  # noqa
                    elif errors == "fix":
                        warnings.warn(f"Ignoring unsupported config parameter '{key}: {config[key]}'.")  # noqa
                        to_del.add(key)
            for key in to_del:
                config.pop(key)
            conf = cls(**config, run_validate=False)
            conf.validate(errors=errors)
            return conf

    @classmethod
    def write_default_config(cls, outpath):
        cls().save_to_yaml(outpath)

    def save_to_yaml(self, outpath):
        with open(outpath, 'w') as outF:
            for (param_type, param_names) in self.organized_param_names().items():  # noqa
                outF.write(f"# {param_type}\n")
                params_dict = OrderedDict()
                for name in param_names:
                    # Don't want to print out all the weird yaml
                    # encodings for python objects.
                    val = getattr(self, name)
                    if isinstance(val, (OrderedDict, defaultdict)):
                        val = dict(val)
                    params_dict[name] = val
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
