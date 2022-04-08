# These import call the register_dataset and register_sampler hooks in each
# module, populating {DATASET,DATAMODULE,SAMPLER}_LOOKUP
from src.data import n2c2  # noqa F401
from src.data import combined
from src.data.utils import DATASET_LOOKUP, DATAMODULE_LOOKUP, SAMPLER_LOOKUP  # noqa F401


def load_datamodule_from_config(config, errors="raise", **override_kwargs):
    if len(override_kwargs) > 0:
        config = config.copy()
    for (k, v) in override_kwargs.items():
        config.update(k, v, errors=errors)
    datamodule_cls = DATAMODULE_LOOKUP[config.dataset_name]
    datamodule = datamodule_cls.from_config(config)
    if len(config.auxiliary_data) > 0:
        all_datamods = [datamodule]
        dm_names = [datamodule.name]
        for (datakey, kwargs) in config.auxiliary_data.items():
            datamodule_cls = DATAMODULE_LOOKUP[kwargs["dataset_name"]]
            dm = datamodule_cls.from_config(config, **kwargs)
            if dm.name in dm_names:
                raise ValueError(f"Already loaded a datamodule '{dm.name}'!")
            all_datamods.append(dm)
            dm_names.append(dm.name)
        datamodule = combined.CombinedDataModule(
            all_datamods,
            dataset_sample_strategy=config.dataset_sample_strategy,
            dataset_sampler_kwargs=config.dataset_sampler_kwargs)
    return datamodule
