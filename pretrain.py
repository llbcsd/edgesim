import argparse
import logging
import math
import os
import random
import datasets

is_debug = False

if is_debug:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs as DDPK
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    DataCollatorForLanguageModeling
)
from transformers.models.bert.modeling_bert import BertForMaskedLM
import wandb
import sys
import json
from model_utils import BertForMLMAndBRP


logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ARCH_ID_MAP = {
    'mips-32': 2,
    'mips-64': 2, 
    'arm-32': 1, 
    'arm-64': 1, 
    'x86-32': 0, 
    'x86-64': 0,
}


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--train_file", type=str, default='/path/to/train_file', help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='/path/to/tokenizer',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--rela_tokenizer_name",
        type=str,
        default='/path/to/rela_tokenizer',
        help="Pretrained control transfer tokenizer name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", type=str, default='/path/to/output_dir', help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=2023, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default='bert',
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=1,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default='epoch',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--data_cache_dir", type=str, default='/path/to/cache',
        help="Path to the dataset"
    )

    args = parser.parse_args()
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")
    return args


def main():
    args = parse_args()
    kwargs = DDPK(broadcast_buffers=False, find_unused_parameters=False)

    accelerator = Accelerator(mixed_precision='fp16', kwargs_handlers=[kwargs])
    # accelerator = Accelerator(cpu=True)

    project_name = 'cross_arch_ida_edge_bert'
    group_name = 'pretrain_bert'
    experiment_name = args.output_dir.split('/')[-1]
    if accelerator.is_main_process and args.with_tracking:
        wandb.init(project=project_name, group=group_name, name=experiment_name)
        wandb.config.update(args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if os.path.exists(os.path.join(args.data_cache_dir, experiment_name)):
            args.overwrite_cache = False
        else:
            os.makedirs(os.path.join(args.data_cache_dir, experiment_name), exist_ok=True)
    accelerator.wait_for_everyone()

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    
    # load data from txt
    read_file = data_files["train"]

    # debug use
    if is_debug:
        args.config_name = '~/bcsd/bert_torch/hugging_pretrain/config_384.json'    
        args.rela_tokenizer_name = '~/bcsd/datasets/edge_gnn_datas/rela_tokenizer'

    with open(read_file, 'r') as f:
        json_str = f.read()
    parse_json = json.loads(json_str)
    data_list = parse_json['train']
    data_list = random.sample(data_list, 4000000)
    # data_list = data_list[:math.ceil(len(data_list)/4)] # split dataset
    train_list = Dataset.from_list(data_list[math.ceil(len(data_list)*args.validation_split_percentage*0.01):])
    eval_list = Dataset.from_list(data_list[:math.ceil(len(data_list)*args.validation_split_percentage*0.01)])
    raw_datasets = DatasetDict({
        'train': train_list,
        'validation': eval_list,
    })

    del json_str, parse_json, data_list

    # load config
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # load tokenizer & map dataset
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False, do_lower_case=False, do_basic_tokenize=False)
    logger.info("Training new model from scratch")
    column_names = raw_datasets["train"].column_names

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            max_seq_length = 1024
    else:
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    config.vocab_size = tokenizer.vocab_size
    config.max_position_embeddings = max_seq_length

    padding = "max_length" if args.pad_to_max_length else False

    
    rela_tokenizer = AutoTokenizer.from_pretrained(args.rela_tokenizer_name, use_fast=False, do_lower_case=False, do_basic_tokenize=False)

    def tokenize_function(examples):
        result = tokenizer(
            # examples['sentence'],
            examples['0'],
            examples['2'],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
        result['rela_token'] = rela_tokenizer.convert_tokens_to_ids(examples['rela'])
        # result['rela_token_idx'] = [i + 1 for i in examples['sep']] # use [SEP] as predict token
        result['arch_ids'] = [[ARCH_ID_MAP[i]] * max_seq_length for i in examples['arch']]
        return result

    with accelerator.main_process_first():
        raw_datasets = raw_datasets.filter(lambda x: (len(x['0']) + len(x['2']) <= max_seq_length-3))

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on dataset line_by_line",
        )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    if len(train_dataset) > 3:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    # model init 
    model = BertForMLMAndBRP(config, rela_tokenizer.vocab_size)
    model.resize_token_embeddings(len(tokenizer))
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    eval_loss = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            if accelerator.is_main_process and args.with_tracking:
                wandb.log(
                    {
                        "lr": get_lr(optimizer),
                        "train_loss": loss.item(),
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )
            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        correct = 0
        length = 0

        brp_len = 0
        brp_correct = 0

        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            labels = batch.labels
            logits = outputs.logits
            preds = torch.argmax(logits, -1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            length += len(preds)
            correct += (preds == labels).float().sum()
            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
            # brp acc cal
            brp_logits = outputs.brp_logits
            brp_preds = torch.argmax(brp_logits, -1)
            brp_len += len(brp_preds)
            brp_correct += (brp_preds == batch.rela_token).float().sum()

        accuracy = 100 * correct / length
        brp_acc = 100 * brp_correct / brp_len
        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} accuracy: {accuracy}, brp_acc: {brp_acc}")

        if accelerator.is_main_process and args.with_tracking:
            wandb.log(
                {
                    "accuracy": accuracy,
                    "brp_acc": brp_acc,
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "avg_train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
