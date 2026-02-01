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
    lora_r: int = field(default=32, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=64, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    target_modules: str = field(
        default="q_proj,v_proj", metadata={"help": "Which module will be added the lora adapter."}
    )
    lora: bool = field(default=False, metadata={"help": "Whether to apply lora or not."})
    train_on_inputs: bool = field(default=True)
    do_test: bool = field(default=False)
    nncf_config: Optional[str] = field(default=None, metadata={"help": "Path to NNCF config JSON file"})


@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(default=None, metadata={"help": "The path of the dataset to use."})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    val_set_size: int = field(default=120)
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximal total input sequence length after tokenization."},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite cached preprocessed datasets or not."})
    pad_to_max_length: bool = field(default=True, metadata={"help": "Pad to max length or not"})
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

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    set_seed(training_args.seed)

    # ================== Load Model ==================
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )

    nncf_config = None
    compression_ctrl = None

    # Apply NNCF before LoRA
    if training_args.nncf_config is not None:
        nncf_config = NNCFConfig.from_json(training_args.nncf_config)
        if nncf_config.get("log_dir") is None:
            nncf_config["log_dir"] = training_args.output_dir
        if not os.path.exists(training_args.output_dir) and training_args.local_rank in [-1, 0]:
            os.makedirs(nncf_config["log_dir"])

        nncf_network = create_nncf_network(model, nncf_config)
        algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
        compression_ctrl, model = create_compressed_model_from_algo_names(nncf_network, nncf_config, algo_names=[algo_name])

    # Then apply LoRA
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

    # ================== Tokenizer ==================
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # ================== Data & Tokenization ==================
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, truncation=True, max_length=data_args.cutoff_len, padding=True, return_tensors=None)
        if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < data_args.cutoff_len and add_eos_token:
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_prompt(data_point):
        if data_point["input"]:
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
    if training_args.do_train:
        data = load_dataset("json", data_files=data_args.dataset_path)
        val_set_size = data_args.val_set_size
        if val_set_size > 0:
            train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            train_dataset = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            eval_dataset = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        else:
            train_dataset = data["train"].shuffle().map(generate_and_tokenize_prompt)
            eval_dataset = None

    # ================== Trainer ==================
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

    model.config.use_cache = False

    # ================== Training ==================
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = data_args.max_train_samples or len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ================== Evaluate / Test ==================
    if training_args.do_eval:
        eval_results = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples or len(eval_dataset)
        eval_results["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)

    # ================== Test Subnetwork / Search ==================
    def test_subnetwork(model, tokenizer, data_args, dataset_name, compression_ctrl):
        model.eval()
        dataset = load_dataset(dataset_name)
        test_dataset = dataset["test"]
        results = []
        for i, data_point in enumerate(test_dataset):
            prompt = generate_prompt(data_point)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({"instruction": data_point["instruction"], "output": decoded})
        return results

    def search(model, tokenizer, data_args, compression_ctrl):
        search_algo = BaseSearchAlgorithm(compression_ctrl)
        search_algo.search()
        best_subnetwork = search_algo.get_best_subnetwork()
        logger.info("Best subnetwork found via search.")

    if training_args.do_test:
        for dataset_name in TEST_DATASETS:
            logger.info(f"Testing on {dataset_name}...")
            results = test_subnetwork(model, tokenizer, data_args, dataset_name, compression_ctrl)
            result_file = os.path.join(training_args.output_dir, f"{dataset_name}_results.json")
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)

    if nncf_config is not None:
        search(model, tokenizer, data_args, compression_ctrl)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
