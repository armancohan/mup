import argparse
import os
from datetime import datetime
from posixpath import dirname

import pytorch_lightning as pl
import torch
from pydantic.utils import deep_update
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import Config
from data import MupDataModule
from model import Summarizer
from summarizer_trainer import SummarizationModel
from utils.checkpointing import CustomCheckpointIO
from utils.utils import ParseKwargs, SaveHFCallback

# def get_transformer(config):
#     # tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
#     # tokenizer.add_tokens([config.section_sep_token])
#     # tokenizer.model_max_length = config.max_seq_len
#     # hf_model = Summarizer(config, tokenizer)
#     trainer_model = SummarizationModel(config)
#     # model = AutoModelForSeq2SeqLM.from_pretrained(config.tokenizer, low_cpu_mem_usage=True)
#     return trainer_model


def main(config, override_config):
    """
    Trains the model
    :param config:
    :return:
    """
    if config.do_train:
        model = SummarizationModel(config)
    elif config.do_predict:
        assert config.checkpoint_path is not None, "Please provide a checkpoint path with `checkpoint_path` argument``  "
        override_kwargs = override_config.to_kwargs()
        model = SummarizationModel.load_from_checkpoint(config.checkpoint_path, output_dir=override_kwargs["output_dir"])
        # override config
        # because of nested dict, we have to manually override the config. Note that hparams won't work
        orig_config = model.config.to_kwargs()
        updated_config = deep_update(orig_config, override_kwargs)
        updated_config = Config.load_config_from_json(updated_config)
        model.config = updated_config
    data_module = MupDataModule(config, model.tokenizer, config.data_args.dataset_name, config.data_args.test_dataset_name)
    logger = TensorBoardLogger(config.exp_dir, name="log")

    pl.seed_everything(config.seed)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.exp_dir,
        filename="{step}-{vloss:.3f}-{rouge1:.3f}-{rouge2:.3f}",
        save_last=True,
        monitor="rouge2",
        mode="max",
        save_top_k=config.trainer_args.save_top_k,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor()
    save_hf_callback = SaveHFCallback(s3_dest=None, ckpt_dir=config.exp_dir)
    # custom_checkpoint_io = CustomCheckpointIO()

    num_steps = config.trainer_args.num_steps

    # get gradient accumlation steps
    if config.total_batch_size is not None:
        if config.compute_strategy == "ddp":
            dist_world_size = torch.cuda.device_count()
        else:
            dist_world_size = 1
        per_device_batch_size = config.data_args.batch_size
        gradient_accumulation_steps = config.total_batch_size // per_device_batch_size // dist_world_size
        print("\n------------------------------------------")
        print(f"world size: {dist_world_size}, per device batch size: {per_device_batch_size}")
        print(f"Based on total batch size, gradient accumulation steps is set to: {gradient_accumulation_steps}")
        print(f"total training steps: {num_steps}")
        print("\n------------------------------------------")
    else:
        gradient_accumulation_steps = config.trainer_args.pop("gradient_accumulation_steps")

    strategy = DDPStrategy(find_unused_parameters=False) if config.compute_strategy == "ddp" else None

    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        precision=config.compute_precision,
        amp_backend="native",
        # plugins=[custom_checkpoint_io],
        strategy=strategy,
        logger=logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, lr_monitor, save_hf_callback],
        max_steps=num_steps,
        min_steps=num_steps,
        val_check_interval=config.trainer_args.pop("val_check_interval"),
        accumulate_grad_batches=gradient_accumulation_steps,
        limit_val_batches=config.trainer_args.pop("limit_val_batches"),
    )
    if config.do_train:
        trainer.fit(model, data_module)
    if config.do_predict:
        trainer.predict(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)
    override_config = Config(kwargs=args.kwargs, set_defaults=False)

    print(f"Start experiment {config.exp_name}")
    # Setup config
    assert config.compute_strategy in ["none", "ddp", "deepspeed_stage_3_offload", "deepspeed_stage_3"]

    print(config.to_json())

    if config.allow_skip_exp and os.path.exists(config.finish_flag_file):
        print(f"Skip finished experiment {config.exp_name}")
    else:
        print(f"Mark experiment {config.exp_name} as claimed")
        with open(config.finish_flag_file, "a+") as f:
            f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + "\n")
        main(config, override_config)
