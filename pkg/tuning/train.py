import functools
import json
import logging
import math
import os
import sys
from types import MethodType

import numpy as np
from dataclasses import dataclass
from typing import Sequence, Union, Tuple, Dict, List, Any, Generator, Optional

import ray
import ray.data
import torch
import evaluate
from pandas import DataFrame
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from ray.air import ScalingConfig, RunConfig
from ray.train import Checkpoint
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from ray.train.torch import TorchTrainer, get_device, TorchConfig
from torch import nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, \
    AutoConfig, AutoModelForCausalLM, Seq2SeqTrainingArguments, BitsAndBytesConfig
from ray.train.huggingface import TransformersTrainer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import load_dataset, Dataset

from callback import LogCallback
from parser import get_train_args
from template import get_template_and_fix_tokenizer
from trainer import SFTTrainer

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# formatter = logging.Formatter(
#     fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S"
# )
# handler = logging.StreamHandler(sys.stdout)
# handler.setFormatter(formatter)
#
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.addHandler(handler)

cpus_per_worker = 8
IGNORE_INDEX = -100
cutoff_len = 1024


def rename_columns(batch: DataFrame, columns):
    return batch.rename(columns=columns)


def preprocess_dataset(
        dataset: Union["Dataset", "IterableDataset"],
        tokenizer: "PreTrainedTokenizer",
        training_args: "Seq2SeqTrainingArguments"
) -> Union["Dataset", "IterableDataset"]:
    template = get_template_and_fix_tokenizer("llama2", tokenizer)

    def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
        for i in range(len(examples["instruction"])):
            query, response = examples["instruction"][i], examples["response"][i]
            query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
            history = examples["history"][i] if "history" in examples else None
            system = examples["system"][i] if "system" in examples else None
            yield query, response, history, system

    def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        # print(examples)

        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for query, response, history, system in construct_example(examples):
            if not (isinstance(query, str) and isinstance(response, str) and query != "" and response != ""):
                continue

            input_ids, labels = [], []
            for turn_idx, (source_ids, target_ids) in enumerate(template.encode_multiturn(
                    tokenizer, query, response, history, system
            )):
                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(cutoff_len * (len(source_ids) / total_len))
                max_target_len = int(cutoff_len * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len:
                    target_ids = target_ids[:max_target_len]

                if turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

            if template.efficient_eos:
                input_ids += [tokenizer.eos_token_id]
                labels += [tokenizer.eos_token_id]

            if len(input_ids) > cutoff_len:
                input_ids = input_ids[:cutoff_len]
                labels = labels[:cutoff_len]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        ))

    preprocess_func = preprocess_supervised_dataset
    print_function = print_supervised_dataset_example
    new_dataset = dataset.map_batches(preprocess_func)
    if training_args.should_log:
        try:
            print_function(new_dataset.take(1)[0])
        except StopIteration:
            raise RuntimeError("Empty dataset!")
    return new_dataset


def trainer_init_per_worker(config):
    print("--- train_task, pid: ", os.getpid())

    cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print("local_rank:", local_rank)
    device_id = cuda_visible_device[local_rank]
    print("device_id:", device_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    torch.cuda.set_device(int(device_id))

    # device setting
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.current_device())
    device_ids = torch._utils._get_all_device_indices()
    print("device_ids:", device_ids)
    if len(device_ids) <= 0:
        print("invalid device_ids, exit")
        return

    training_args = config.get("training_args", None)
    finetuning_args = config.get("finetuning_args", None)
    model_args = config.get("model_args", None)
    data_args = config.get("data_args", None)
    tokenizer = config.get("tokenizer", None)

    # read dataset
    train_ds = ray.train.get_dataset_shard("train")
    print(f"train_ds: {train_ds}")

    def train_gen():
        for row in train_ds.iter_rows():
            yield row

    train_dataset = Dataset.from_generator(train_gen)
    print(train_dataset)
    print('------')
    print(train_dataset[0])

    eval_ds = ray.train.get_dataset_shard("evaluation")
    print(f"eval_ds: {eval_ds}")

    def eval_gen():
        for row in eval_ds.iter_rows():
            yield row

    eval_dataset = None
    evaluation_strategy = "no"
    if eval_ds:
        eval_dataset = Dataset.from_generator(eval_gen)
        print(eval_dataset)
        evaluation_strategy = "steps"

    train_ds_len = len(list(train_ds.iter_batches(batch_size=1)))
    steps_per_epoch = math.ceil(train_ds_len / training_args.per_device_train_batch_size)
    print(f"train_ds_len: {train_ds_len}, steps_per_epoch: {steps_per_epoch}")

    new_training_args = Seq2SeqTrainingArguments(
        training_args.output_dir,
        logging_steps=10,
        save_strategy="no",
        evaluation_strategy=evaluation_strategy,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        warmup_steps=training_args.warmup_steps,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        optim=training_args.optim,
        lr_scheduler_type=training_args.lr_scheduler_type,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        push_to_hub=False,
        report_to="none",
        disable_tqdm=False,  # declutter the output a little
        fp16=training_args.fp16,
        gradient_checkpointing=True,
        deepspeed=training_args.deepspeed,
        log_level="info",
    )

    print(f"new_training_args: {new_training_args}".replace("\n", " "))

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    compute_dtype = getattr(config, "torch_dtype", None)

    if model_args.quantization == "int4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    elif model_args.quantization == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
    )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # turn off when gradient checkpointing is enabled
    print("Gradient checkpointing enabled.")

    output_layer_name = "lm_head"

    if hasattr(model, output_layer_name):
        output_layer = getattr(model, output_layer_name)
        if isinstance(output_layer, torch.nn.Linear):
            def forward_in_fp32(self, x: torch.Tensor) -> torch.Tensor:
                return output_layer.__class__.forward(self, x.to(output_layer.weight.dtype)).to(torch.float32)

            output_layer.forward = MethodType(forward_in_fp32, output_layer)

    target_modules = finetuning_args.lora_target

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetuning_args.lora_rank,
        lora_alpha=finetuning_args.lora_alpha,
        lora_dropout=finetuning_args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=finetuning_args.additional_target
    )
    model = get_peft_model(model, lora_config)
    if id(model.peft_config) != id(model.base_model.peft_config):  # https://github.com/huggingface/peft/issues/923
        model.base_model.peft_config = model.peft_config
    model.train()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=4,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX
    )

    trainer = SFTTrainer(
        model=model,
        args=new_training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[LogCallback(metrics_export_address=finetuning_args.metrics_export_address, uid=finetuning_args.uid)],
    )

    trainer = prepare_trainer(trainer)
    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)

    checkpoint = None
    if ray.train.get_context().get_world_rank() == 0:
        checkpoint = Checkpoint.from_directory(training_args.output_dir)
    ray.train.report(metrics=train_result.metrics, checkpoint=checkpoint)


def main():
    print("init")
    ray.init()

    training_args, finetuning_args, model_args, data_args = get_train_args()

    print(f"training_args: {training_args}".replace("\n", " "))
    print(finetuning_args)
    print(model_args)
    print(data_args)

    model_path = model_args.model_name_or_path
    use_gpu = True
    num_workers = finetuning_args.num_workers

    if data_args.block_size > 0:
        global cutoff_len
        cutoff_len = data_args.block_size

    # read dataset
    print("preprocess_dataset")
    columns_map = {
        "instruction": "instruction",
        "output": "response"
    }
    if data_args.columns:
        print(data_args.columns)
        columns_map.update({v: k for k, v in json.loads(data_args.columns).items()})

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    train_dataset = ray.data.read_csv(data_args.train_path). \
        map_batches(rename_columns, fn_args=[columns_map], batch_format="pandas")
    print(train_dataset)
    train_dataset = preprocess_dataset(train_dataset, tokenizer, training_args)

    input_datasets = {"train": train_dataset}

    if data_args.evaluation_path:
        evaluation_dataset = ray.data.read_csv(data_args.train_path). \
            map_batches(rename_columns, fn_args=[columns_map], batch_format="pandas")
        print(evaluation_dataset)
        evaluation_dataset = preprocess_dataset(evaluation_dataset, tokenizer, training_args)
        input_datasets["evaluation"] = evaluation_dataset

    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu,
                                   resources_per_worker={"GPU": 1, "CPU": cpus_per_worker},
                                   trainer_resources={"GPU": 0}
                                   )

    ray_trainer = TorchTrainer(
        train_loop_per_worker=trainer_init_per_worker,
        train_loop_config={
            "training_args": training_args,
            "finetuning_args": finetuning_args,
            "model_args": model_args,
            "data_args": data_args,
            "tokenizer": tokenizer,
        },
        scaling_config=scaling_config,
        datasets=input_datasets,
        run_config=RunConfig(
            storage_path=finetuning_args.storage_path,
            # checkpoint_config=ray.train.CheckpointConfig(
            #     num_to_keep=1,
            #     checkpoint_score_attribute="eval_loss",
            #     checkpoint_score_order="min",
            # ),
        )
    )
    result = ray_trainer.fit()
    checkpoint_path = result.checkpoint.path

    print(f"result path {checkpoint_path}")

    file_path = "/home/ray/checkpoint_path"

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(checkpoint_path)


if __name__ == '__main__':
    main()
