from .n2c2 import n2c2DataModule
from .combined import CombinedDataModule


DATAMODULE_LOOKUP = {
    "n2c2Context": n2c2DataModule,
    "n2c2Assertion": n2c2DataModule,
    "i2b2Event": n2c2DataModule,
}
