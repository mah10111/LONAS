#!/usr/bin/env python
# coding=utf-8
"""
run_commonsense.py - Robust final version which ensures NNCF runs on a clean base model
(no PEFT/LoRA wrappers). After compression, LoRA may be attached/loaded again.
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

# NNCF imports (assume nncf installed / available)
from nncf import NNCFConfig
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
    create_compressed_model_from_algo_names,
)
from nncf.experimental.torch.nas.bootstrapNAS import BaseSearchAlgorithm
from nncf.torch.model_creation import create_nncf_network

check_min_version("4.31.0")

# ----- logging -----
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
# Helpers for PEFT/LoRA and cleaning
# -------------------------------
def contains_lora_modules(mod: torch.nn.Module) -> bool:
    """Heuristic to detect presence of LoRA adapters in a model's modules."""
    for name, module in mod.named_modules():
        nl = name.lower()
        if "lora" in nl or "lora_" in nl:
            return True
        # some wrappers include ModuleDict('lora_A') in repr
        try:
            if "ModuleDict" in repr(type(module)) and "lora" in repr(module).lower():
                return True
        except Exception:
            pass
    return False


def merge_peft_if_possible(peft_model: PeftModel):
    """Attempt to merge PEFT adapters into base model (merge_and_unload -> get_base_model -> base_model)."""
    if hasattr(peft_model, "merge_and_unload") and callable(getattr(peft_model, "merge_and_unload")):
        try:
            logger.info("Calling merge_and_unload() on PeftModel...")
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
        except Exception as e:
            logger.warning("get_base_model() failed: %s", e)
    if hasattr(peft_model, "base_model"):
        logger.info("Using .base_model attribute fallback.")
        return peft_model.base_model
    # fallback to original
    logger.warning("Could not merge/unload PeftModel; returning original object as fallback.")
    return peft_model


def aggressive_clean_lora_from_module(module, modulename):
    """
    Try to remove common LoRA attributes/children from a module (best-effort).
    Returns True if something removed.
    """
    removed_any = False
    # common attribute names across PEFT versions
    lora_attr_candidates = ["lora_A", "lora_B", "lora_WA", "lora_WB", "lora_dropout", "lora_scaling", "lora_alpha"]
    for attr in lora_attr_candidates:
        if hasattr(module, attr):
            try:
                delattr(module, attr)
                logger.info("Removed attribute %s from module %s", attr, modulename)
                removed_any = True
            except Exception as e:
                logger.debug("Failed to delattr %s from %s: %s", attr, modulename, e)
    # remove named children starting with lora
    try:
        for child_name, _ in list(module.named_children()):
            if child_name.lower().startswith("lora"):
                try:
                    if hasattr(module, "__delattr__"):
                        delattr(module, child_name)
                        logger.info("Deleted child %s from %s", child_name, modulename)
                        removed_any = True
                except Exception as e:
                    logger.debug("Failed to delete child %s from %s: %s", child_name, modulename, e)
    except Exception:
        pass
    return removed_any


def get_clean_base_model_for_nncf(original_model: torch.nn.Module, model_args, offload_folder: str):
    """
    Return a model suitable to pass to create_nncf_network/create_compressed_model_from_algo_names.
    Strategy:
      - If original_model is PeftModel: try to merge_and_unload or get_base_model.
      - If candidate still contains LoRA: load a fresh base model from model_args.model_name_or_path.
      - If user provided lora_weights and we can merge them onto fresh base, optionally merge temporarily to produce base.
    """
    candidate = original_model

    try:
        if isinstance(candidate, PeftModel):
            logger.info("Original model is PeftModel — attempting to merge/unload before NNCF.")
            candidate = merge_peft_if_possible(candidate)
    except Exception as e:
        logger.warning("isinstance(PeftModel) check or merge attempt raised: %s", e)

    # fallback attribute checks
    if not isinstance(candidate, PeftModel):
        if hasattr(candidate, "base_model"):
            candidate = candidate.base_model
        elif hasattr(candidate, "model"):
            candidate = candidate.model

    # if no lora -> good
    if not contains_lora_modules(candidate):
        logger.info("Candidate model is clean (no LoRA detected).")
        return candidate

    logger.warning("Detected LoRA modules in candidate model. Loading fresh base model for NNCF.")
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
        logger.error("Failed to load fresh base model for fallback: %s", e)
        # fallback to best-effort candidate (may fail)
        return candidate

    # optional: if user provided lora_weights and we can attach and merge them into fresh base, do it to get merged result
    if getattr(model_args, "lora_weights", None):
        try:
            logger.info("Attaching provided LoRA weights to fresh base to attempt merge (temporary).")
            tmp = PeftModel.from_pretrained(fresh_base, model_args.lora_weights, torch_dtype=torch.float16, device_map="auto")
            merged_tmp = merge_peft_if_possible(tmp)
            if merged_tmp is not None and not contains_lora_modules(merged_tmp):
                logger.info("Successfully merged LoRA into fresh base and obtained clean model.")
                return merged_tmp
            # otherwise fallback to fresh_base without LoRA
            logger.warning("Merged fresh tmp still contains LoRA or merge returned None; using fresh base without LoRA.")
            return fresh_base
        except Exception as e:
            logger.warning("Failed to attach/merge provided LoRA weights on fresh base: %s. Using fresh base.", e)
            return fresh_base
    else:
        return fresh_base


# -------------------------------
# dataclasses (arguments)
# -------------------------------
@dataclass
class LonasTrainingArguments(TrainingArguments):
    nncf_config: Optional[str] = field(
        default=None, metadata={"help": "Path to NNCF config file for NAS/quantization"}
    )
    lora_r: int = field(default=32, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=64, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    target_modules: str = field(default="q_proj,v_proj", metadata={"help": "Which module will be added the lora adapter."})
    lora: bool = field(default=False, metadata={"help": "Whether to apply lora or not."})
    train_on_inputs: bool = field(default=True)
    do_test: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(default=None, metadata={"help": "The path of the dataset to use."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use."})
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

    # logging config
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logger.setLevel(training_args.get_process_log_level())

    # checkpoint detection
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) exists and is not empty.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    set_seed(training_args.seed)

    # ensure offload folder exists for device_map="auto"
    offload_folder = os.path.join(training_args.output_dir, "offload")
    os.makedirs(offload_folder, exist_ok=True)

    # ----------------------------
    # 1) Load user-visible model (may be wrapped by PEFT)
    # ----------------------------
    logger.info("Loading model (user-facing) from %s ...", model_args.model_name_or_path)
    user_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder=offload_folder,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )

    # ----------------------------
    # 2) Prepare NNCF config (if provided)
    # ----------------------------
    nncf_config = None
    if training_args.nncf_config:
        logger.info("Loading NNCF config from %s", training_args.nncf_config)
        nncf_config = NNCFConfig.from_json(training_args.nncf_config)
        if nncf_config.get("log_dir") is None:
            nncf_config["log_dir"] = training_args.output_dir
        os.makedirs(nncf_config["log_dir"], exist_ok=True)

    compression_ctrl = None
    model_after_compression = None

    # ----------------------------
    # 3) Apply NNCF on a CLEAN base model
    # ----------------------------
    if nncf_config is not None:
        logger.info("Preparing clean base model for NNCF...")
        clean_base = get_clean_base_model_for_nncf(user_model, model_args, offload_folder)

        # sanity debug: print whether any lora-like modules remain
        if contains_lora_modules(clean_base):
            logger.error("Clean base STILL contains LoRA modules. Aborting NNCF to avoid graph mismatch.")
            # dump details for debugging
            logger.error("Detailed suspicious modules in clean_base:")
            for n, m in clean_base.named_modules():
                if "lora" in n.lower() or "ModuleDict" in repr(type(m)) and "lora" in repr(m).lower():
                    logger.error("MODULE: %s | type=%s | repr_preview=%s", n, type(m), repr(m)[:300])
            raise RuntimeError("Clean base model contains LoRA-like modules; cannot proceed with NNCF.")

        logger.info("Creating NNCF network from clean base and applying compression...")
        try:
            nncf_network = create_nncf_network(clean_base, nncf_config)
            algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
            compression_ctrl, model_after_compression = create_compressed_model_from_algo_names(
                nncf_network, nncf_config, algo_names=[algo_name]
            )
            logger.info("NNCF compression applied successfully.")
        except Exception as e:
            logger.exception("Error while applying NNCF compression: %s", e)
            raise

    # ----------------------------
    # 4) Continue with compressed model (if created) or user's model
    # ----------------------------
    model = model_after_compression if model_after_compression is not None else user_model

    # ----------------------------
    # 5) Attach or load LoRA after compression (if user requested)
    # ----------------------------
    if training_args.lora:
        if model_args.lora_weights is None:
            logger.info("Attaching fresh LoRA adapters on post-NNCF model...")
            lora_cfg = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                lora_dropout=training_args.lora_dropout,
                target_modules=training_args.target_modules.split(","),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
        else:
            logger.info("Loading LoRA weights from %s onto current model...", model_args.lora_weights)
            model = PeftModel.from_pretrained(model, model_args.lora_weights, torch_dtype=torch.float16, device_map="auto")

    # ----------------------------
    # 6) Tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    logger.info("Model and tokenizer ready — proceeding to dataset & trainer pipeline.")

    # ----------------------------
    # 7) Dataset/tokenization/helpers (kept from original)
    # ----------------------------
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=data_args.cutoff_len,
            padding=True,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < data_args.cutoff_len
            and add_eos_token
        ):
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
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
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
            eval_dataset = None

    # ----------------------------
    # 8) Trainer
    # ----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if getattr(training_args, "do_eval", False) else None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compression_ctrl=compression_ctrl,
    )

    if nncf_config is not None:
        if not (training_args.local_rank in [-1, 0] or training_args.no_cuda):
            compression_ctrl.distributed()

    model.config.use_cache = False

    # ----------------------------
    # 9) Training loop
    # ----------------------------
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ----------------------------
    # 10) (Optional) evaluation/test/search blocks can be re-used from your original script.
    # If you want, I can append the evaluate_one_sample, evaluate, test_subnetwork and search flows back in.
    # ----------------------------

    kwargs = {"finetuned_from": model_args.model_name_or_path}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
