import csv
import functools
import json
import os
import pathlib
import random
import re
from dataclasses import dataclass, field
from multiprocessing.spawn import prepare
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from rouge_score import rouge_scorer
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_url
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    get_linear_schedule_with_warmup,
)
from transformers.models.bart.modeling_bart import shift_tokens_right

from model import Summarizer
from utils.get_optimizer import get_optimizer
from utils.get_scheduler import get_scheduler

random.seed(2)


class SummarizationModel(LightningModule):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
        tokenizer.add_tokens([config.section_sep_token])
        self.tokenizer = tokenizer
        section_sep_token_id = self.tokenizer.convert_tokens_to_ids([self.config.section_sep_token])
        assert len(section_sep_token_id) == 1
        self.section_sep_token_id = section_sep_token_id[0]

        self.model = Summarizer(config, tokenizer)

        # if config.gradient_checkpointing and (
        #     "led" in config.model_name_or_path or "primera" in config.model_name_or_path.lower()
        # ):
        #     import ipdb

        #     ipdb.set_trace()

    def prepare_input(self, batch):
        """get the attention mask for encoder and decoder input ids"""
        inputs = {
            "input_ids": batch["input_ids"].input_ids,
            "attention_mask": batch["input_ids"].attention_mask,
            "ids": batch["paper_ids"],
        }
        if (
            "led" in self.config.model_name_or_path or "primera" in self.config.model_name_or_path.lower()
        ) and self.config.use_global_attention:
            inputs["global_attention_mask"] = inputs["input_ids"].new_zeros(
                inputs["input_ids"].shape[0], inputs["input_ids"].shape[1]
            )
            inputs["global_attention_mask"][:, 0] = 1
            inputs["global_attention_mask"] = torch.where(
                inputs["input_ids"] == self.section_sep_token_id, 1, inputs["global_attention_mask"]
            )
        if "decoder_input_ids" in batch:
            if batch["decoder_input_ids"].input_ids[0][0] != self.model.model.config.decoder_start_token_id:
                # add decoder start token id
                inputs["decoder_input_ids"] = shift_tokens_right(
                    batch["decoder_input_ids"].input_ids,
                    self.model.model.config.pad_token_id,
                    self.model.model.config.decoder_start_token_id,
                )
                labels = batch["decoder_input_ids"].input_ids.clone()
                inputs["labels"] = labels
                # inputs["labels"] = [
                # [-100 if token == self.tokenizer.pad_token_id else token for token in labels]
                # for labels in batch["labels"]
                # ]
            assert labels.shape[1] == inputs["decoder_input_ids"].shape[1]
        else:
            inputs["decoder_input_ids"] = torch.full(
                (inputs["input_ids"].shape[0], 1), self.model.model.config.decoder_start_token_id
            )
            labels = None
        return inputs

    def forward(self, **inputs):
        # print("doc:")
        # print(self.tokenizer.decode(inputs["input_ids"][0]))
        # print("\n\n\nref:")
        # print(self.tokenizer.decode(inputs["labels"][0]))
        # print("-------")
        # import ipdb

        # ipdb.set_trace()

        outputs = self.model(**inputs)
        lm_logits = outputs.get("logits")

        # Same behavior as modeling_bart.py, besides ignoring pad_token_id
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        assert lm_logits.shape[-1] == self.model.model.config.vocab_size
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), inputs["labels"].view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        batch = self.prepare_input(batch)
        ids = batch.pop("ids")
        loss = self(**batch)
        # self.log("loss", loss)
        self.log(
            "lr",
            torch.tensor(self.trainer.optimizers[0].param_groups[0]["lr"]),
            prog_bar=True,
        )
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0, return_preds=False, compute_metrics=True):
        for p in self.model.parameters():
            p.requires_grad = False
        batch = self.prepare_input(batch)
        ids = batch.pop("ids")
        vloss = self(**batch)
        kwargs = {}
        if "global_attention_mask" in batch:
            kwargs["global_attention_mask"] = batch["global_attention_mask"]

        generated_ids = self.model.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            max_length=self.config.generation.max_generation_length,
            num_beams=self.config.generation.num_beams,
            length_penalty=self.config.generation.length_penalty,
            min_length=self.config.generation.min_generation_length,
            no_repeat_ngram_size=self.config.generation.no_repeat_ngram_size,
            **kwargs,
        )
        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        if compute_metrics:
            gold_str = self.tokenizer.batch_decode(batch["decoder_input_ids"].tolist(), skip_special_tokens=True)
            scorer = rouge_scorer.RougeScorer(rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=False)
            rouge1 = rouge2 = rougel = rougelsum = 0.0
            preds = []
            for _id, ref, pred in zip(ids, gold_str, generated_str):
                if return_preds:
                    preds.append({"id": _id.item(), "ref": ref, "pred": pred})
                score = scorer.score(ref, pred)
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougel += score["rougeL"].fmeasure
                rougelsum += score["rougeLsum"].fmeasure
            rouge1 /= len(generated_str)
            rouge2 /= len(generated_str)
            rougel /= len(generated_str)
            rougelsum /= len(generated_str)

            output = {
                "vloss": vloss,
                "rouge1": vloss.new_zeros(1) + rouge1,
                "rouge2": vloss.new_zeros(1) + rouge2,
                "rougeL": vloss.new_zeros(1) + rougel,
                "rougeLsum": vloss.new_zeros(1) + rougelsum,
                "preds": preds,
            }
        else:
            preds = [{"id": _id, "pred": pred} for _id, pred in zip(ids, generated_str)]
            output = {"preds": preds}
        return output

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx, return_preds=True, compute_metrics=False)

    def on_predict_epoch_end(self, outputs):
        assert len(outputs) == 1
        preds = [output["preds"] for output in outputs[0]]
        # flatten list of lists
        results = [item for sublist in preds for item in sublist]

        pathlib.Path(self.hparams.output_dir).mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank() if self.config.compute_strategy == "ddp" else ""
        with open(self.hparams.output_dir + f"/preds-{rank}.json", "w") as fout:
            json.dump(results, fout)
        results_csv = pd.DataFrame(results)
        with open(self.hparams.output_dir + f"/preds-{rank}.csv", "w") as fout:
            results_csv.to_csv(fout, index=False)
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                all_res = {}
                all_res_csv = []
                for rank in range(torch.distributed.get_world_size()):
                    with open(self.hparams.output_dir + f"/preds-{rank}.json", "r") as fin:
                        results = json.load(fin)
                    results_csv = pd.DataFrame(results)
                    with open(self.hparams.output_dir + f"/preds-{rank}.csv", "w") as fout:
                        results_csv.to_csv(fout, index=False)
                    all_res.update(results)
                    all_res_csv.append(results_csv)
                all_res_csv = pd.concat(all_res_csv)
                all_res_csv.rename({"id": "paper_id", "pred": "summary"}, axis=1, inplace=True)
                with open(self.hparams.output_dir + "/preds.json", "w") as fout:
                    json.dump(all_res, fout)
                with open(self.hparams.output_dir + "/preds.csv", "w") as fout:
                    all_res_csv.to_csv(fout, index=False)
                print("Predictions saved to:", self.hparams.output_dir)
            torch.distributed.barrier()
        return {}

    def validation_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True

        names = ["vloss", "rouge1", "rouge2", "rougeL", "rougeLsum"]
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            if self.config.compute_strategy == "ddp":
                torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric /= self.trainer.world_size
            metrics.append(metric)
        logs = dict(zip(*[names, metrics]))
        logs["step"] = torch.tensor(self.global_step, dtype=torch.long, device=logs["vloss"].device)
        for k, v in logs.items():
            self.log(k, v)
        return {"avg_val_loss": logs["vloss"], "log": logs, "progress_bar": logs}

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        if self.config.trainer_args.num_steps:
            self.total_steps = self.config.trainer_args.num_steps
        elif self.trainer.max_epochs > 0:
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
            tb_size = self.config.data_args.batch_size * max(1, self.trainer.gpus)
            ab_size = self.trainer.accumulate_grad_batches
            self.total_steps = (len(train_loader.dataset) * float(self.trainer.max_epochs) // tb_size) // ab_size
            assert self.total_steps > 0
        else:
            raise ValueError("Either num_steps or max_epochs must be specified")

    def configure_optimizers(self):
        optimizer, self.trainable_param_names = get_optimizer(self.model, self.config.trainer_args)
        scheduler = get_scheduler(optimizer, self.config.trainer_args, self.total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
