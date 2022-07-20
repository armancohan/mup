import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LEDForConditionalGeneration,
    MBartTokenizer,
    MBartTokenizerFast,
)
from transformers.modeling_outputs import Seq2SeqModelOutput

logger = logging.getLogger(__name__)


class Summarizer(nn.Module):
    def __init__(self, config, tokenizer) -> None:
        super().__init__()
        transformer_config = AutoConfig.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir,
            gradient_checkpointing=getattr(config, "gradient_checkpointing", None),
        )
        if "led-" in config.model_name_or_path.lower():
            model = LEDForConditionalGeneration.from_pretrained(
                config.model_name_or_path,
                config=transformer_config,
                cache_dir=config.cache_dir,
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config.model_name_or_path,
                config=transformer_config,
                cache_dir=config.cache_dir,
            )

        model.resize_token_embeddings(len(tokenizer))

        if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[config.data_args.lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(config.data_args.lang)

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < config.data_args.max_source_length
        ):
            if getattr(model.config, "max_encoder_position_embeddings", 0) >= config.data_args.max_source_length:
                pass
            elif config.resize_position_embeddings is None:
                logger.warning(
                    f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                    f"to {config.data_args.max_source_length}."
                )
                model.resize_position_embeddings(config.data_args.max_source_length)
            elif config.resize_position_embeddings:
                model.resize_position_embeddings(config.data_args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {config.data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                    f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                    "resize the model's position encodings by passing `--resize_position_embeddings`."
                )

        self.model = model

    def forward(self, **kwargs) -> Union[Tuple, Seq2SeqModelOutput]:
        return self.model(**kwargs)

    def save_pretrained(self, save_directory, **kwargs) -> None:
        self.model.save_pretrained(save_directory, **kwargs)

    def from_pretrained(self, path, **kwargs) -> None:
        self.model.from_pretrained(path, **kwargs)
