import json
from typing import Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

SECTION_SEP_TOKEN = "<sec/>"


class MUPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        split=None,
        section_sep_token=SECTION_SEP_TOKEN,
        include_abstract=False,
        predict_mode=False,
        use_only_one_summary=False,
    ) -> None:
        super().__init__()
        if dataset_name.endswith(".jsonl"):
            with open(dataset_name, "r") as fin:
                self.data = [json.loads(line) for line in fin]
        else:
            self.data = load_dataset(dataset_name)
        if split is not None:
            self.data = self.data[split]
        self.tokenizer = tokenizer
        self.section_sep_token = section_sep_token
        self.include_abstract = include_abstract
        self.predict_mode = predict_mode
        self.use_only_one_summary = use_only_one_summary

    def __len__(self):
        return len(self.data)

    def get_text_from_sections(self, sections: List[Dict[str, str]], abstract: Optional[str] = None) -> str:
        """get a flat text from list of sections"""
        text = ""
        if self.include_abstract:
            text = abstract + f" {self.section_sep_token}"
        for section in sections:
            section_text = section["text"]
            text += section_text + f" {self.section_sep_token}"
        return text

    def __getitem__(self, idx):
        entry = self.data[idx]
        text = self.get_text_from_sections(entry["sections"], entry["abstractText"])
        paper_id = entry["paper_id"]
        summaries = entry.get("summaries") or None
        title = entry.get("title") or ""
        summary_index = 0
        if summaries is None:
            output = [
                {
                    "text": text,
                    "paper_id": paper_id + f"___{summary_index}",
                    "summary": None,
                    "title": title,
                }
            ]
        else:
            output = []
            if self.predict_mode or self.use_only_one_summary:
                summaries = [summaries[0]]
            for summary_index, summary in enumerate(summaries):
                output.append(
                    {
                        "text": text,
                        "paper_id": paper_id + (f"___{summary_index}" if not self.predict_mode else ""),
                        "summary": summary,
                        "title": title,
                    }
                )
        return output


class MUPDatasetNested(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name,
        tokenizer,
        data_split,
        section_sep_token=SECTION_SEP_TOKEN,
        include_abstract=False,
        config=None,
        predict_mode=False,
    ) -> None:
        super().__init__()
        dataset = MUPDataset(
            dataset_name,
            tokenizer,
            data_split,
            section_sep_token,
            include_abstract,
            predict_mode,
            config.use_only_one_summary,
        )
        self.tokenizer = dataset.tokenizer
        # flatten the dataset (dataset is a list of lists)
        self.data = [item for example in tqdm(dataset, desc="loading dataset...") for item in example]
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids = self.tokenizer(
            [example["text"] for example in batch],
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        if batch[0].get("summary"):
            decoder_input_ids = self.tokenizer(
                [example["summary"] for example in batch],
                return_tensors="pt",
                truncation=True,
                padding=True,
                add_special_tokens=False,
                max_length=self.config.max_decoder_length,
            )
        paper_ids = [example["paper_id"] for example in batch]
        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "paper_ids": paper_ids,
        }


class MupDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer, dataset_name, test_dataset_name=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.test_dataset_name = test_dataset_name

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train_dataset = MUPDatasetNested(
            self.dataset_name,
            self.tokenizer,
            "train",
            self.config.section_sep_token,
            self.config.include_abstract,
            self.config,
        )
        self.validation_dataset = MUPDatasetNested(
            self.dataset_name,
            self.tokenizer,
            "validation",
            self.config.section_sep_token,
            self.config.include_abstract,
            self.config,
        )
        if self.test_dataset_name is not None:
            self.test_dataset = MUPDatasetNested(
                self.test_dataset_name,
                self.tokenizer,
                None,
                self.config.section_sep_token,
                self.config.include_abstract,
                self.config,
                predict_mode=True,
            )
        print(f"Train size {len(self.train_dataset)}")
        print(f"Eval size {len(self.validation_dataset)}")
        if self.test_dataset_name is not None:
            print(f"Test size {len(self.test_dataset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.data_args.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.config.trainer_args.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.config.data_args.eval_batch_size,
            shuffle=False,
            collate_fn=self.validation_dataset.collate_fn,
            num_workers=self.config.trainer_args.num_workers,
        )

    def test_dataloader(self):
        if self.test_dataset_name is None:
            return None
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.data_args.eval_batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=self.config.trainer_args.num_workers,
        )

    def predict_dataloader(self):
        if self.test_dataset_name is None:
            return None
        print(len(self.test_dataset))
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.data_args.eval_batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=self.config.trainer_args.num_workers,
        )
