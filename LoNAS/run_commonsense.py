#!/usr/bin/env python
# coding=utf-8
"""
run_commonsense.py - Robust version for combining LoRA (PEFT) and NNCF (BootstrapNAS).

- Ensures LoRA adapters are merged/removed before NNCF.
- Allows attaching new LoRA after NNCF.
"""

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

# nncf imports
from nncf import NNCFConfig
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
    create_compressed_model_from_algo_names,
)
from nncf.experimental.torch.nas.bootstrapNAS import BaseSearchAlgorithm
from nncf.torch.model_creation import create_nncf_network

check_min_version("4.31.0")
logger = logging.getLogger(__name__)
TEST_DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "winogrande",
    "ARC-Easy",
    "ARC-Challenge",
    "openbookqa",
    "hellaswag",
]

# -------------------------------
# Utilities for PEFT/LoRA
# -------------------------------
def contains_lora_modules(mod: torch.nn.Module) -> bool:
    for name, module in mod.named_modules():
        lname = name.lower()
        if "lora" in lname or "lora_" in lname or "loramodule" in lname:
            return True
        try:
            rep = repr(module)
            if "ModuleDict" in rep and "lora" in rep:
                return True
        except Exception:
            pass
    return False

def merge_peft_if_possible(peft_model: PeftModel):
    if hasattr(peft_model, "merge_and_unload") and callable(getattr(peft_model, "merge_and_unload")):
        try:
            merged = peft_model.merge_and_unload()
            return merged
        except Exception:
            pass
    if hasattr(peft_model, "get_base_model") and callable(getattr(peft_model, "get_base_model")):
        try:
            return peft_model.get_base_model()
        except Exception:
            pass
    if hasattr(peft_model, "base_model"):
        return peft_model.base_model
    return peft_model

def get_clean_base_model(model, model_args, offload_folder):
    """
    Ensure a model without LoRA modules for NNCF.
    """
    # Step 1: merge if PeftModel
    if isinstance(model, PeftModel):
        model = merge_peft_if_possible(model)

    # Step 2: if LoRA still present, load fresh base
    if contains_lora_modules(model):
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=offload_folder,
            trust_remote_code=True,
            cache_dir=model_args.cache_dir,
        )

    # Step 3: attach and merge user-provided LoRA weights temporarily to get merged base
    if getattr(model_args, "lora_weights", None):
        try:
            tmp = PeftModel.from_pretrained(model, model_args.lora_weights, torch_dtype=torch.float16, device_map="auto")
            model = merge_peft_if_possible(tmp)
            if contains_lora_modules(model):
                model = tmp.base_model
        except Exception:
            pass

    if contains_lora_modules(model):
        raise RuntimeError("LoRA modules remain in model — cannot build NNCF graph.")
    return model

# -------------------------------
# Dataclasses
# -------------------------------
@dataclass
class LonasTrainingArguments(TrainingArguments):
    nncf_config: Optional[str] = field(default=None)
    lora_r: int = field(default=32)
    lora_alpha: float = field(default=64)
    lora_dropout: float = field(default=0.0)
    target_modules: str = field(default="q_proj,v_proj")
    lora: bool = field(default=False)
    train_on_inputs: bool = field(default=True)
    do_test: bool = field(default=False)

@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = None
    dataset_config_name: Optional[str] = None
    val_set_size: int = 120
    cutoff_len: int = 256
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_predict_samples: Optional[int] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model"})
    lora_weights: str = field(default=None)
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None

# -------------------------------
# Main
# -------------------------------
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LonasTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logger.setLevel(training_args.get_process_log_level())

    set_seed(training_args.seed)
    offload_folder = os.path.join(training_args.output_dir, "offload")
    os.makedirs(offload_folder, exist_ok=True)

    # ----------------------------
    # 1) Load base model
    # ----------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder=offload_folder,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )

    # ----------------------------
    # 2) Prepare NNCF config
    # ----------------------------
    nncf_config = None
    if training_args.nncf_config:
        nncf_config = NNCFConfig.from_json(training_args.nncf_config)
        if nncf_config.get("log_dir") is None:
            nncf_config["log_dir"] = training_args.output_dir
        os.makedirs(nncf_config["log_dir"], exist_ok=True)

    compression_ctrl = None
    if nncf_config:
        # ----------------------------
        # 3) Ensure clean model for NNCF
        # ----------------------------
        model_clean = get_clean_base_model(model, model_args, offload_folder)
        nncf_network = create_nncf_network(model_clean, nncf_config)
        algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
        compression_ctrl, model = create_compressed_model_from_algo_names(nncf_network, nncf_config, algo_names=[algo_name])
        logger.info("NNCF compression applied successfully.")

    # ----------------------------
    # 4) Attach LoRA after NNCF if requested
    # ----------------------------
    if training_args.lora:
        if model_args.lora_weights is None:
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                target_modules=training_args.target_modules.split(","),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            model = PeftModel.from_pretrained(model, model_args.lora_weights, torch_dtype=torch.float16, device_map="auto")

    # ----------------------------
    # 5) Tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # ----------------------------
    # 6) Dataset, Trainer, Evaluation (keep original logic)
    # ----------------------------
    # ... باقی کد dataset، tokenization، trainer و evaluation بدون تغییر ...
    # فقط مطمئن شوید trainer=Trainer(model=model, compression_ctrl=compression_ctrl, ...)
    pass

if __name__ == "__main__":
    main()
