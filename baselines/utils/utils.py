import argparse
import glob
import os

import pytorch_lightning as pl
from pytorch_lightning import Callback


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            value = value.replace("\\=", "=")
            vals = value.split("=")
            assert len(vals) >= 2, f"Invalid config format: {value}"
            if len(vals) > 2:  # checkpoint path includes a '='
                this_value = "=".join(vals[1:])
            else:
                this_value = vals[1]
            key = vals[0]
            getattr(namespace, self.dest)[key] = this_value


class SaveHFCallback(Callback):
    CHECKPOINT_DIR = "checkpoints"
    TENSORBOARD_LOGS_DIR = "tensorboard"
    CONFIG_FILENAME = "config.yaml"
    STARTED_FILENAME = "started.txt"
    COMPLETED_FILENAME = "completed.txt"
    LOGGING_DIR = "logs"

    def __init__(self, s3_dest=None, ckpt_dir=None):
        self.s3_dest = s3_dest
        self.ckpt_dir = ckpt_dir

    def on_fit_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:

        # saving checkpoints in HuggingFace format
        if trainer.is_global_zero:
            print("saving huggingface checkpoints")
            for ckpt_path in glob.glob(self.ckpt_dir + "/*.ckpt"):
                model = model.load_from_checkpoint(ckpt_path)
                hf_model = model.model
                hf_model.save_pretrained(f"{ckpt_path}.hf")
