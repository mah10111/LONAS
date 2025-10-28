#!/usr/bin/env python
# coding=utf-8
import copy
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

from nncf import NNCFConfig
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
    create_compressed_model_from_algo_names,
)
from nncf.experimental.torch.nas.bootstrapNAS import BaseSearchAlgorithm
from nncf.torch.model_creation import create_nncf_network

check_min_version("4.31.0")
logger = logging.getLogger(__name__)
TEST_DATASETS = ["boolq", "piqa", "social_i_qa", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa", "hellaswag"]


@dataclass
class LonasTrainingArguments(TrainingArguments):
    nncf_config: Optional[str] = field(default=None, metadata={"help": "Path to NNCF config file for NAS/quantization"})
    lora_r: int = field(default=32, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=64, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    target_modules: str = field(default="q_proj,v_proj", metadata={"help": "Which module will be added the lora adapter."})
    lora: bool = field(default=False, metadata={"help": "Whether to apply lora or not."})
    train_on_inputs: bool = field(default=True)
    do_test: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(default=None, metadata={"help": "The path of the dataset to use."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    val_set_size: int = field(default=120)
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)
    pad_to_max_length: bool = field(default=True)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    cutoff_len: int = field(default=256)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    lora_weights: str = field(default=None)
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where to store downloaded pretrained models"})
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=False)


def unwrap_peft(model):
    """Unwrap PEFT/LoRA to get base model for NNCF."""
    if hasattr(model, "base_model"):
        return model.base_model
    if hasattr(model, "model"):
        return model.model
    return model


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LonasTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    set_seed(training_args.seed)

    # ----------------------------
    # Load base model
    # ----------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )

    # ----------------------------
    # NNCF Config
    # ----------------------------
    nncf_config = None
    compression_ctrl = None
    if training_args.nncf_config is not None:
        nncf_config = NNCFConfig.from_json(training_args.nncf_config)
        nncf_config["log_dir"] = nncf_config.get("log_dir", training_args.output_dir)
        os.makedirs(nncf_config["log_dir"], exist_ok=True)

        logger.info("Creating NNCF network and applying compression...")
        nncf_network = create_nncf_network(model, nncf_config)
        algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
        nncf_network = unwrap_peft(nncf_network)
        compression_ctrl, model = create_compressed_model_from_algo_names(
            nncf_network, nncf_config, algo_names=[algo_name]
        )

        # Debug module names
        try:
            print("=== module names containing 'gate' or 'proj' (after NNCF) ===")
            for n, m in model.named_modules():
                if any(k in n for k in ['gate', 'proj', 'q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj']):
                    print(n, type(m))
            print("=== end module list ===")
        except Exception as e:
            logger.warning(f"Failed to print module names for debug: {e}")

    # ----------------------------
    # Add LoRA (after NNCF)
    # ----------------------------
    if training_args.lora:
        if model_args.lora_weights is None:
            logger.info("Adding LoRA modules...")
            config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                target_modules=training_args.target_modules.split(","),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            model.print_trainable_parameters()
        else:
            logger.info("Loading LoRA modules...")
            model = PeftModel.from_pretrained(model, model_args.lora_weights, torch_dtype=torch.float16, device_map="auto")

    # ----------------------------
    # Load tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # ----------------------------
    # Train / Eval datasets
    # ----------------------------
    train_dataset, eval_dataset = None, None
    if training_args.do_train or training_args.do_search:
        data = load_dataset("json", data_files=data_args.dataset_path)
        val_set_size = data_args.val_set_size
        if val_set_size > 0:
            train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            train_dataset = train_val["train"].shuffle()
            eval_dataset = train_val["test"].shuffle()
        else:
            train_dataset = data["train"].shuffle()

    # ----------------------------
    # Trainer
    # ----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compression_ctrl=compression_ctrl,
    )

    logger.info("==== Training/Evaluation Completed Successfully ====")


if __name__ == "__main__":
    main()
