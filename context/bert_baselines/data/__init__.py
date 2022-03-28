# These import call the register_dataset and register_sampler hooks in head
# module, populating {DATASET,DATAMODULE,SAMPLER}_LOOKUP
from . import n2c2
from . import combined
from .utils import DATASET_LOOKUP, DATAMODULE_LOOKUP, SAMPLER_LOOKUP
