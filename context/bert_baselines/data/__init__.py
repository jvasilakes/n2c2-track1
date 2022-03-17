from .n2c2 import n2c2ContextDataModule, n2c2AssertionDataModule
from .combined import CombinedDataModule


DATAMODULE_LOOKUP = {
    "n2c2Context": n2c2ContextDataModule,
    "n2c2Assertion": n2c2AssertionDataModule,
}
