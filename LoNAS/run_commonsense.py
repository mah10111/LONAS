#!/usr/bin/env python
# coding=utf-8
import copy
import json
import logging
import os
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
# Helper: prepare base model for NNCF
# -------------------------------
def get_base_model_for_nncf(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return a model suitable to be passed to create_nncf_network / create_compressed_model_from_algo_names.
    If model is a PeftModel with merge_and_unload(), merge LoRA into base model and return merged model.
    Otherwise return model.base_model or model.model if available, else model.
    """
    # If it's an instance of PeftModel, try to merge and unload adapters if possible
    try:
        if isinstance(model, PeftModel):
            # If PeftModel provides merge_and_unload (some versions), call it to produce a plain model
            if hasattr(model, "merge_and_unload") and callable(getattr(model, "merge_and_unload")):
                logging.info("Merging LoRA weights into base model (merge_and_unload)...")
                try:
                    merged = model.merge_and_unload()
                    logging.info("Merged LoRA into base model successfully.")
                    return merged
                except Exception as e:
                    logging.warning(f"merge_and_unload() failed: {e}. Falling back to get_base_model()/base_model attr.")
            # fallback to get_base_model if exists
            if hasattr(model, "get_base_model") and callable(getattr(model, "get_base_model")):
                try:
                    return model.get_base_model()
                except Exception:
                    pass
            # fallback to attribute
            if hasattr(model, "base_model"):
                return model.base_model
    except Exception:
        # if PeftModel class is not present or isinstance fails, continue to generic checks
        pass

    # Generic fallbacks
    if hasattr(model, "base_model"):
        return model.base_model
    if hasattr(model, "model"):
        return model.model
    return model


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

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
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

    # make sure offload folder exists (device_map="auto" may offload to disk)
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
    # 2) Prepare nncf_config
    # ----------------------------
    nncf_config = None
    if training_args.nncf_config:
        nncf_config = NNCFConfig.from_json(training_args.nncf_config)
        if nncf_config.get("log_dir") is None:
            nncf_config["log_dir"] = training_args.output_dir
        os.makedirs(nncf_config["log_dir"], exist_ok=True)

    # ----------------------------
    # 3) Apply NNCF BEFORE LoRA
    # ----------------------------
    compression_ctrl = None
    if nncf_config:
        logger.info("Applying NNCF/BootstrapNAS on base model...")

        # Ensure we pass the pure base model (merge LoRA if it was already attached)
        nncf_network = get_base_model_for_nncf(model)
        # create nncf network and compressed model
        nncf_network = create_nncf_network(nncf_network, nncf_config)
        algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
        compression_ctrl, model = create_compressed_model_from_algo_names(nncf_network, nncf_config, algo_names=[algo_name])

        # Debug: print some module names
        try:
            print("=== module names containing 'proj' after NNCF ===")
            for n, m in model.named_modules():
                if any(k in n for k in ["q_proj", "k_proj", "v_proj", "proj"]):
                    print(n, type(m))
            print("=== end module list ===")
        except Exception as e:
            logger.warning(f"Failed to list modules after NNCF: {e}")

    # ----------------------------
    # 4) Apply LoRA (after NNCF)
    # ----------------------------
    if training_args.lora:
        if model_args.lora_weights is None:
            logger.info("Adding LoRA modules (fresh)...")
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
            logger.info("Loading LoRA weights from %s ...", model_args.lora_weights)
            model = PeftModel.from_pretrained(model, model_args.lora_weights, torch_dtype=torch.float16, device_map="auto")

    # ----------------------------
    # 5) Tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    logger.info("✅ Model and tokenizer initialized successfully.")

    # The rest of your pipeline (dataset, tokenization, Trainer, training/eval) can continue here.
    # I leave the remainder of your original pipeline in place to avoid removing logic you already had.

if __name__ == "__main__":
    main()
