DATASET_LOOKUP = {}
DATAMODULE_LOOKUP = {}
SAMPLER_LOOKUP = {}


def register_dataset(dataset_name, datamodule):
    def add_to_lookup(cls):
        DATASET_LOOKUP[dataset_name] = cls
        DATAMODULE_LOOKUP[dataset_name] = datamodule
        return cls
    return add_to_lookup


def register_sampler(sampler_name):
    def add_to_lookup(cls):
        SAMPLER_LOOKUP[sampler_name] = cls
        return cls
    return add_to_lookup
