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
    """Heuristic to detect presence of LoRA adapters in a model's modules."""
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
    """
    If peft_model supports merge_and_unload(), call it and return merged base model.
    Fallback to get_base_model() or .base_model attribute.
    """
    # prefer merge_and_unload if available
    if hasattr(peft_model, "merge_and_unload") and callable(getattr(peft_model, "merge_and_unload")):
        logger.info("Calling merge_and_unload() on PeftModel...")
        try:
            merged = peft_model.merge_and_unload()
            logger.info("merge_and_unload() succeeded.")
            return merged
        except Exception as e:
            logger.warning("merge_and_unload() failed: %s", e)

    # fallback to get_base_model()
    if hasattr(peft_model, "get_base_model") and callable(getattr(peft_model, "get_base_model")):
        try:
            base = peft_model.get_base_model()
            logger.info("Used get_base_model() fallback.")
            return base
        except Exception:
            logger.warning("get_base_model() fallback failed.")

    # fallback to attribute .base_model
    if hasattr(peft_model, "base_model"):
        logger.info("Using .base_model attribute fallback.")
        return peft_model.base_model

    # as last resort return original object
    logger.warning("Could not merge/unload PeftModel; returning original object as fallback.")
    return peft_model


def get_clean_base_model_for_nncf(original_model: torch.nn.Module, model_args, offload_folder: str):
    """
    Try to obtain a model without LoRA modules to pass to NNCF.
    Strategy:
    1) If original_model already PeftModel: try merge_and_unload/get_base_model/.base_model
    2) If still contains lora modules => load fresh base from model_args.model_name_or_path and:
        a) if model_args.lora_weights provided => attach then merge (if possible)
        b) else use fresh base
    3) If still LoRA present after all fallbacks => raise RuntimeError with helpful message
    """
    candidate = original_model

    # 1) If it's a PeftModel or wrapped, attempt to merge/unload / get base
    try:
        if isinstance(candidate, PeftModel):
            candidate = merge_peft_if_possible(candidate)
    except Exception as e:
        logger.warning("isinstance(PeftModel) check or merge attempt raised: %s", e)

    # Generic attribute fallbacks
    if not isinstance(candidate, PeftModel):
        if hasattr(candidate, "base_model"):
            candidate = candidate.base_model
        elif hasattr(candidate, "model"):
            candidate = candidate.model

    # If no LoRA present -> good
    if not contains_lora_modules(candidate):
        logger.info("Candidate clean model detected (no LoRA modules).")
        return candidate

    # 2) Robust fallback: load fresh base from model_args.model_name_or_path
    logger.warning("LoRA modules detected in candidate model. Attempting robust fallback by loading fresh base model.")
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
        # if we cannot load fresh base, return the best-effort candidate (NNCF will likely fail)
        return candidate

    # If user provided lora weights, try to attach and merge them to fresh base (so we can produce merged result)
    if getattr(model_args, "lora_weights", None):
        try:
            logger.info("Attaching provided LoRA weights to fresh base and attempting merge...")
            tmp = PeftModel.from_pretrained(fresh_base, model_args.lora_weights, torch_dtype=torch.float16, device_map="auto")
            merged_tmp = merge_peft_if_possible(tmp)
            if merged_tmp is not None:
                # merged_tmp could be a base model or fallback; ensure it is not containing LoRA
                if not contains_lora_modules(merged_tmp):
                    logger.info("Successfully obtained merged fresh model without LoRA.")
                    return merged_tmp
                else:
                    logger.warning("Merged fresh model still contains LoRA modules after merge attempt.")
                    return getattr(tmp, "base_model", fresh_base)
            else:
                logger.warning("merge_peft_if_possible returned None; using fresh_base.")
                return fresh_base
        except Exception as e:
            logger.warning("Failed to attach/merge provided LoRA weights onto fresh base: %s. Using fresh base without LoRA.", e)
            return fresh_base
    else:
        # no lora_weights provided, use fresh base
        return fresh_base


# -------------------------------
# Dataclasses (arguments)
# -------------------------------
@dataclass
class LonasTrainingArguments(TrainingArguments):
    nncf_config: Optional[str] = field(default=None, metadata={"help": "Path to NNCF config file for NAS/quantization"})
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

    # detect last checkpoint if training
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) exists and is not empty.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    set_seed(training_args.seed)

    # ensure offload folder exists (for device_map="auto")
    offload_folder = os.path.join(training_args.output_dir, "offload")
    os.makedirs(offload_folder, exist_ok=True)

    # ----------------------------
    # 1) Load base model (may be wrapped)
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
    # 2) Prepare nncf_config (if any)
    # ----------------------------
    nncf_config = None
    if training_args.nncf_config:
        logger.info("Loading NNCF config from %s", training_args.nncf_config)
        nncf_config = NNCFConfig.from_json(training_args.nncf_config)
        if nncf_config.get("log_dir") is None:
            nncf_config["log_dir"] = training_args.output_dir
        os.makedirs(nncf_config["log_dir"], exist_ok=True)

    # ----------------------------
    # 3) Ensure clean base for NNCF (merge LoRA if needed)
    # ----------------------------
    compression_ctrl = None
    if nncf_config:
        logger.info("Preparing clean base model for NNCF (merging/removing LoRA if present)...")
        nncf_candidate = get_clean_base_model_for_nncf(model, model_args, offload_folder)

        # final check
        if contains_lora_modules(nncf_candidate):
            logger.error("LoRA modules remain in the candidate model after all attempts. NNCF cannot proceed.")
            raise RuntimeError(
                "LoRA modules remain in the model candidate — cannot build NNCF graph. "
                "Ensure LoRA is merged (merge_and_unload) or call NNCF on the original base model."
            )

        # create nncf network and compressed model
        try:
            nncf_network = create_nncf_network(nncf_candidate, nncf_config)
            algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
            compression_ctrl, model = create_compressed_model_from_algo_names(nncf_network, nncf_config, algo_names=[algo_name])
            logger.info("NNCF compression applied successfully.")
        except Exception as e:
            logger.exception("Failed while creating compressed model from NNCF: %s", e)
            raise

        # debug print some modules
        try:
            print("=== module names containing 'proj' after NNCF ===")
            for n, m in model.named_modules():
                if any(k in n for k in ["q_proj", "k_proj", "v_proj", "proj"]):
                    print(n, type(m))
            print("=== end module list ===")
        except Exception:
            logger.debug("Failed to print modules after NNCF (non-fatal).")

    # ----------------------------
    # 4) Apply or load LoRA (after NNCF)
    # ----------------------------
    if training_args.lora:
        if model_args.lora_weights is None:
            logger.info("Adding new LoRA modules on top of (possibly compressed) model...")
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
            logger.info("Loading LoRA weights from %s onto current model...", model_args.lora_weights)
            model = PeftModel.from_pretrained(model, model_args.lora_weights, torch_dtype=torch.float16, device_map="auto")

    # ----------------------------
    # 5) Tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    logger.info("Model & tokenizer ready. Proceeding to dataset & training pipeline...")

    # ----------------------------
    # 6) Dataset + tokenization helpers (kept same as your original)
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
    if training_args.do_train or training_args.do_search:
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
    # 7) Trainer
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

    if nncf_config is not None:
        if not (training_args.local_rank in [-1, 0] or training_args.no_cuda):
            compression_ctrl.distributed()

    model.config.use_cache = False

    # ----------------------------
    # 8) Training loop
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
    # 9) Simple evaluate/predict helpers (kept)
    # ----------------------------
    def extract_answer(dataset_name, sentence: str) -> str:
        sentence_ = sentence.strip()
        if dataset_name == "boolq":
            pred_answers = re.findall(r"true|false", sentence_)
            return pred_answers[0] if pred_answers else ""
        if dataset_name == "piqa":
            pred_answers = re.findall(r"solution1|solution2", sentence_)
            return pred_answers[0] if pred_answers else ""
        if dataset_name in ["social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"]:
            pred_answers = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence_)
            return pred_answers[0] if pred_answers else ""
        if dataset_name == "hellaswag":
            pred_answers = re.findall(r"ending1|ending2|ending3|ending4", sentence_)
            return pred_answers[0] if pred_answers else ""
        if dataset_name == "winogrande":
            pred_answers = re.findall(r"option1|option2", sentence_)
            return pred_answers[0] if pred_answers else ""
        return ""

    def load_test_data(test_dataset) -> list:
        file_path = f"datasets/{test_dataset}/test.json"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"can not find dataset file : {file_path}")
        json_data = json.load(open(file_path, "r"))
        return json_data

    def generate_prompt_eval(instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Response:
"""

    def evaluate_one_sample(
        instruction,
        input=None,
        model=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=32,
        **kwargs,
    ):
        prompts = generate_prompt_eval(instruction, input)
        inputs = tokenizer(prompts, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()

    def evaluate(model_, dataset_name, save_file):
        model_.eval()
        dataset = load_test_data(dataset_name)

        total = len(dataset)
        correct = 0
        output_data = []
        for idx, data in enumerate(dataset):
            instruction = data.get("instruction")
            output = evaluate_one_sample(instruction, model=model_)
            label = data.get("answer")
            flag = False
            predict = extract_answer(dataset_name, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data["output_pred"] = output
            new_data["pred"] = predict
            new_data["flag"] = flag
            output_data.append(new_data)
            print(data["instruction"])
            print(output)
            print("prediction:", predict)
            print("label:", label)

            print(f"\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}")

            with open(save_file, "w+") as f:
                json.dump(output_data, f, indent=4)

        acc = correct / total
        return acc

    # ----------------------------
    # 10) do_test / search flows (kept original logic)
    # ----------------------------
    if training_args.do_test and training_args.local_rank <= 0:
        if compression_ctrl is not None:
            trainer.compression_ctrl.multi_elasticity_handler.enable_all()
            compression_ctrl.multi_elasticity_handler.width_handler.width_num_params_indicator = -1
            heuristic_config = {
                k: v[(len(v) - 1) // 2] for k, v in compression_ctrl.multi_elasticity_handler.width_search_space.items()
            }
            heuristic_config = {ElasticityDim.WIDTH: heuristic_config}
            trainer.compression_ctrl.multi_elasticity_handler.activate_subnet_for_config(heuristic_config)
            # Evaluate heuristic subnetwork as desired...
        else:
            all_results = []
            for test_dataset in TEST_DATASETS:
                logger.info(f"*** Evaluation on {test_dataset} ***")
                save_file = os.path.join(training_args.output_dir, f"{test_dataset}.res.json")
                non_zero_params = sum([(param.data != 0).sum().item() for _, param in trainer.model.named_parameters()])
                accuracy = evaluate(trainer.model, test_dataset, save_file)
                all_results.append(accuracy)
                metrics = {
                    f"{test_dataset}_accuracy": accuracy,
                    "non_zero_params": non_zero_params,
                }
                trainer.save_metrics("eval", metrics)
            avg_metrics = {"avg_accuracy": sum(all_results) / len(all_results)}
            trainer.save_metrics("eval", avg_metrics)
            trainer.log_metrics("eval", avg_metrics)

    if training_args.do_search and nncf_config is not None and training_args.local_rank <= 0:
        logger.info("*** Search ***")
        trainer.compression_ctrl.multi_elasticity_handler.enable_all()
        search_algo = BaseSearchAlgorithm.from_config(trainer.model, trainer.compression_ctrl, nncf_config)

        def validate_model_fn(model_, eval_dataset):
            correct = 0
            for data in eval_dataset:
                instruction = data.get("instruction")
                output = evaluate_one_sample(instruction, model=model_)
                label = data.get("answer")
                dataset_name = None
                if label in ["true", "false"]:
                    dataset_name = "boolq"
                elif "solution" in label:
                    dataset_name = "piqa"
                elif "answer" in label:
                    dataset_name = "social_i_qa"
                elif "ending" in label:
                    dataset_name = "hellaswag"
                elif "option" in label:
                    dataset_name = "winogrande"
                predict = extract_answer(dataset_name, output)
                if label == predict:
                    correct += 1
            acc = correct / len(eval_dataset)
            return acc

        trainer.compression_ctrl.multi_elasticity_handler.activate_supernet()
        max_eval_acc = validate_model_fn(trainer.model, eval_dataset)

        compression_ctrl.multi_elasticity_handler.width_handler.width_num_params_indicator = -1
        heuristic_config = {k: v[(len(v) - 1) // 2] for k, v in compression_ctrl.multi_elasticity_handler.width_search_space.items()}
        heuristic_config = {ElasticityDim.WIDTH: heuristic_config}
        trainer.compression_ctrl.multi_elasticity_handler.activate_subnet_for_config(heuristic_config)
        heu_eval_acc = validate_model_fn(trainer.model, eval_dataset)

        metrics = {"val_maximal_accuracy": max_eval_acc, "val_heuristic_accuracy": heu_eval_acc}
        trainer.save_metrics("eval", metrics)
        trainer.log_metrics("eval", metrics)

        elasticity_ctrl, best_config, performance_metrics = search_algo.run(
            validate_model_fn, eval_dataset, training_args.output_dir
        )

        search_algo.search_progression_to_csv()
        search_algo.evaluators_to_csv()
        search_algo.visualize_search_progression()

        logger.info("Best config: {best_config}".format(best_config=best_config))
        logger.info("Performance metrics: {performance_metrics}".format(performance_metrics=performance_metrics))
        trainer.save_metrics("eval", {"performance_metrics": list(performance_metrics)})

        trainer.compression_ctrl.multi_elasticity_handler.activate_subnet_for_config(best_config)
        best_eval_acc = validate_model_fn(trainer.model, eval_dataset)
        trainer.save_metrics("eval", {"val_best_accuracy": best_eval_acc})

    kwargs = {"finetuned_from": model_args.model_name_or_path}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
