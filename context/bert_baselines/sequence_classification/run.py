import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config import ExperimentConfig
from data import n2c2SentencesDataModule
from model import BertMultiHeadedSequenceClassifier

# TODO: log start/end times
# TODO: give this script "train", "validate", and "predict" modes.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="Path to a yaml config file")
    return parser.parse_args()


def main(config):
    logdir = os.path.join("logs")
    os.makedirs(logdir, exist_ok=True)
    version = get_next_version(logdir, config.name)
    version_dir = os.path.join(logdir, config.name, f"version_{version}")
    os.makedirs(version_dir, exist_ok=False)
    config.save_to_yaml(os.path.join(version_dir, "config.yaml"))

    pl.seed_everything(config.random_seed, workers=True)

    dm = n2c2SentencesDataModule(
            config.data_dir,
            config.sentences_dir,
            batch_size=config.batch_size,
            model_name_or_path=config.model_name_or_path,
            tasks_to_load=config.tasks_to_load,
            max_seq_length=config.max_seq_length,
            window_size=config.window_size,
            )
    dm.setup()

    print("Label Spec")
    print('  ' + str(dm.label_spec))
    model = BertMultiHeadedSequenceClassifier(
            config.model_name_or_path,
            label_spec=dm.label_spec,
            freeze_pretrained=config.freeze_pretrained,
            use_entity_spans=config.use_entity_spans,
            lr=config.lr,
            weight_decay=config.weight_decay,
            )

    logger = TensorBoardLogger(
            save_dir=logdir, version=version, name=config.name)
    checkpoint_cb = ModelCheckpoint(
            monitor="avg_micro_f1",
            mode="max",
            filename="{epoch:02d}-{avg_micro_f1:.2f}-{avg_val_loss:.2f}")
    available_gpus = min(1, torch.cuda.device_count())
    trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.max_epochs,
            gpus=available_gpus,
            gradient_clip_val=config.gradient_clip_val,
            deterministic=True,
            callbacks=[checkpoint_cb],
            )
    trainer.fit(model, datamodule=dm)


# TODO: finish get_next_version
def get_next_version(logdir, config_name):
    return 1


if __name__ == "__main__":
    args = parse_args()
    config = ExperimentConfig.from_yaml_file(args.config_file)
    print(config)
    main(config)
