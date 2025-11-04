#!/usr/bin/env python
# coding=utf-8
"""
run_commonsense.py - Robust version for combining LoRA (PEFT) and NNCF (BootstrapNAS).

Key behavior:
- Ensure LoRA (PEFT) adapters are merged/removed before passing a model to NNCF.
- If user provided separate lora_weights path, attempt to merge them into a fresh base before NNCF.
- Apply NNCF (create_compressed_model_from_algo_names) on a clean base model.
- After compression, optionally attach LoRA (new or loaded) for fine-tuning.
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

# nncf imports (Intel NNCF)
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
# Utilities for dealing with PEFT/LoRA
# -------------------------------
def contains_lora_modules(mod: torch.nn.Module) -> bool:
    """Detect presence of LoRA adapters in a model's modules."""
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
    """Merge PEFT/LoRA weights into base model if possible."""
    if hasattr(peft_model, "merge_and_unload") and callable(getattr(peft_model, "merge_and_unload")):
        logger.info("Calling merge_and_unload() on PeftModel...")
        try:
            merged = peft_model.merge_and_unload()
            logger.info("merge_and_unload() succeeded.")
            return merged
        except Exception as e:
            logger.warning("merge_and_unload() failed: %s", e)

    if hasattr(peft_model, "get_base_model") and callable(getattr(peft_model, "get_base_model")):
        try:
            base = peft_model.get_base_model()
            logger.info("Used get_base_model() fallback.")
            return base
        except Exception:
            logger.warning("get_base_model() fallback failed.")

    if hasattr(peft_model, "base_model"):
        logger.info("Using .base_model attribute fallback.")
        return peft_model.base_model

    logger.warning("Could not merge/unload PeftModel; returning original object as fallback.")
    return peft_model


def get_clean_base_model_for_nncf(original_model: torch.nn.Module, model_args, offload_folder: str):
    """Return a model without LoRA modules to pass to NNCF."""
    candidate = original_model

    # 1) Merge/unload PEFT if possible
    try:
        if isinstance(candidate, PeftModel):
            candidate = merge_peft_if_possible(candidate)
    except Exception as e:
        logger.warning("isinstance(PeftModel) check or merge attempt raised: %s", e)

    # Attribute fallbacks
    if not isinstance(candidate, PeftModel):
        if hasattr(candidate, "base_model"):
            candidate = candidate.base_model
        elif hasattr(candidate, "model"):
            candidate = candidate.model

    if not contains_lora_modules(candidate):
        logger.info("Candidate clean model detected (no LoRA modules).")
        return candidate

    # 2) Load fresh base model
    logger.warning("LoRA modules detected. Loading fresh base model for NNCF...")
    try:
        fresh_base = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=offload_folder,
            trust_remote_code=True,
            cache_dir=model_args.cache_dir,
        )
    except Exception as e:
        logger.error("Failed to load fresh base model: %s", e)
        return candidate

    if getattr(model_args, "lora_weights", None):
        try:
            logger.info("Attaching provided LoRA weights to fresh base...")
            tmp = PeftModel.from_pretrained(fresh_base, model_args.lora_weights, torch_dtype=torch.float16, device_map="auto")
            merged_tmp = merge_peft_if_possible(tmp)
            if merged_tmp is not None and not contains_lora_modules(merged_tmp):
                return merged_tmp
            else:
                return getattr(tmp, "base_model", fresh_base)
        except Exception as e:
            logger.warning("Failed to merge LoRA weights onto fresh base: %s. Using fresh base.", e)
            return fresh_base
    else:
        return fresh_base


# -------------------------------
# Dataclasses (arguments)
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
    dataset_path: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
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
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier"})
    lora_weights: str = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=False)


# -------------------------------
# Main
# -------------------------------
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
    logger.setLevel(training_args.get_process_log_level())
    datasets.utils.logging.set_verbosity(logger.level)
    transformers.utils.logging.set_verbosity(logger.level)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) exists and is not empty.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    set_seed(training_args.seed)
    offload_folder = os.path.join(training_args.output_dir, "offload")
    os.makedirs(offload_folder, exist_ok=True)

    # ----------------------------
    # Load model
    # ----------------------------
    logger.info("Loading model '%s' ...", model_args.model_name_or_path)
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
    # Prepare NNCF
    # ----------------------------
    nncf_config = None
    if training_args.nncf_config:
        logger.info("Loading NNCF config from %s", training_args.nncf_config)
        nncf_config = NNCFConfig.from_json(training_args.nncf_config)
        if nncf_config.get("log_dir") is None:
            nncf_config["log_dir"] = training_args.output_dir
        os.makedirs(nncf_config["log_dir"], exist_ok=True)

    # ----------------------------
    # Prepare clean model for NNCF
    # ----------------------------
    compression_ctrl = None
    if nncf_config:
        nncf_candidate = get_clean_base_model_for_nncf(model, model_args, offload_folder)
        if contains_lora_modules(nncf_candidate):
            raise RuntimeError("LoRA modules remain in the model candidate — cannot build NNCF graph.")

        nncf_network = create_nncf_network(nncf_candidate, nncf_config)
        algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
        compression_ctrl, model = create_compressed_model_from_algo_names(nncf_network, nncf_config, algo_names=[algo_name])

    # ----------------------------
    # Apply or load LoRA
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
    # Tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    logger.info("Model & tokenizer ready.")

    # Dataset helpers
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, truncation=True, max_length=data_args.cutoff_len, padding=True, return_tensors=None)
        if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < data_args.cutoff_len and add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_prompt(data_point):
        if data_point.get("input"):
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point['instruction']}

### Input:
{data_point['input']}

### Response:
{data_point['output']}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{data_point['instruction']}

### Response:
{data_point['output']}"""

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not training_args.train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    train_dataset, eval_dataset = None, None
    if training_args.do_train or getattr(training_args, "do_search", False):
        data = load_dataset("json", data_files=data_args.dataset_path)
        val_set_size = data_args.val_set_size
        if val_set_size > 0:
            train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            train_dataset = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            eval_dataset = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        else:
            train_dataset = data["train"].shuffle().map(generate_and_tokenize_prompt)

    # ----------------------------
    # Trainer
    # ----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if getattr(training_args, "do_eval", False) else None,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        compression_ctrl=compression_ctrl,
    )

    model.config.use_cache = False

    # ----------------------------
    # Training loop
    # ----------------------------
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples else len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
