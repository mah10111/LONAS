# تغییرات کلیدی:
# - حذف `training_args.nncf_config`
# - خواندن مستقیم فایل NNCF config از مسیر ثابت
# - بهبود امنیت مسیر و بارگذاری JSON
# - اطمینان از سازگاری با PyTorch >=2.1 و HuggingFace Transformers >=4.31

# --- ابتدا همان import ها ---
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
    lora_r: int = field(default=32, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=64, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    target_modules: str = field(
        default="q_proj,v_proj", metadata={"help": "Which module will be added the lora adapter."}
    )
    lora: bool = field(default=False, metadata={"help": "Whether to apply lora or not."})
    train_on_inputs: bool = field(default=True)
    do_test: bool = field(default=False)
    nncf_config: Optional[str] = field(
    default=None,
    metadata={"help": "Path to NNCF config JSON file"}
    )

# --- DataTrainingArguments و ModelArguments بدون تغییر ---
@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(default=None)
    val_set_size: int = field(default=120)
    cutoff_len: int = field(default=256)

@dataclass
class ModelArguments:
    model_name_or_path: str
    lora_weights: str = field(default=None)
    cache_dir: Optional[str] = field(default=None)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LonasTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- بارگذاری NNCF config مستقیم ---
    nncf_config_path = "nncf_config/unified_commonsense/nncf_lonas_llama_7b.json"
    if os.path.exists(nncf_config_path):
        with open(nncf_config_path, 'r') as f:
            nncf_config = json.load(f)
    else:
        nncf_config = None
        logger.warning(f"Could not find nncf_config at {nncf_config_path}. Continuing without NNCF.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # --- Set seed ---
    set_seed(training_args.seed)

    # --- Load model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )

    # --- LoRA ---
    if training_args.lora and model_args.lora_weights is None:
        logger.info("adding LoRA modules...")
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
    elif training_args.lora:
        logger.info("Loading LoRA modules...")
        model = PeftModel.from_pretrained(model, model_args.lora_weights, torch_dtype=torch.float16, device_map="auto")

    # --- NNCF / Compression ---
    compression_ctrl = None
    if nncf_config is not None:
        nncf_network = create_nncf_network(model, nncf_config)
        algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
        compression_ctrl, model = create_compressed_model_from_algo_names(
            nncf_network, nncf_config, algo_names=[algo_name]
        )

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # --- Data loading & tokenization ---
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, truncation=True, max_length=data_args.cutoff_len, padding=True, return_tensors=None)
        if result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token and len(result["input_ids"]) < data_args.cutoff_len:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_prompt(data_point):
        if data_point.get("input"):
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

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
            eval_dataset = None

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if getattr(training_args, "do_eval", False) else None,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        compression_ctrl=compression_ctrl,
    )

    # --- Training ---
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or get_last_checkpoint(training_args.output_dir)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
