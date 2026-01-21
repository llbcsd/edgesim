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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)
from transformers.models.bert.modeling_bert import BertModel
import wandb
import torch.nn as nn
from data_utils_ce import FinetuneDatasetCE
from load_data import collate_fn_pair_typeD_with_dir
from model_utils import BertCustomModel
from gcn_edge_model import EdgeSimMPNN
import pickle
import torch.nn.functional as F
import numpy as np
from utils import Similarity


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def get_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    return logger


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--train_file", type=str, default='/path/to/train_file', 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--output_dir", type=str, default='/path/to/output_dir', 
        help="Where to store the final model."
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
        default='/path/to/config.json',
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='/path/to/tokenizer',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='/path/to/pretrain/pytorch_model.bin',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
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
        "--model_type",
        type=str,
        default=None,
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
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
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
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--margin", type=float, default=0.1, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--data_cache_dir", type=str, default='/path/to/cache',
        help="Path to the dataset"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    logger = get_logger()

    project_name = 'cross_arch_ida_edge_bert'
    group_name = 'finetune_bert_ce_EdgeGCN_TypeB_MY_DIR'
    experiment_name = args.output_dir.split('/')[-1]
    if args.with_tracking:
        wandb.init(project=project_name, group=group_name, name=experiment_name)
        wandb.config.update(args)

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.data_cache_dir, experiment_name)):
        args.overwrite_cache = False
    else:
        os.makedirs(os.path.join(args.data_cache_dir, experiment_name), exist_ok=True)

    
    data_files = args.train_file
    with open(data_files, 'rb') as f:
        data =  pickle.load(f)

    device = torch.device('cuda:0')

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False, do_lower_case=False, do_basic_tokenize=False)
    logger.info("Training new model from scratch")
    if args.model_name_or_path is None:
        encode_model = BertModel(config)
    else:
        encode_model = BertCustomModel.from_pretrained(
            args.model_name_or_path, 
            config=config,
            add_pooling_layer=False,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            custom_config={'arch_num':3}
        )
    encode_model.resize_token_embeddings(len(tokenizer))

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            max_seq_length = 1024
    else:
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    
    if is_debug:
        data = random.sample(data, 100)
        args.validation_split_percentage = 10
        args.per_device_train_batch_size = 16
        args.per_device_eval_batch_size = 8

    train_eval_split_index = int(len(data)*args.validation_split_percentage*0.01)
    train_data = data[train_eval_split_index:]
    eval_data = data[:train_eval_split_index]

    train_dataset = FinetuneDatasetCE(train_data, encode_model, device, tokenizer, max_seq_length, is_tensor=True)

    eval_dataset = FinetuneDatasetCE(eval_data, encode_model, device, tokenizer, max_seq_length, is_tensor=True)

    if len(train_dataset) > 3:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn_pair_typeD_with_dir, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn_pair_typeD_with_dir, batch_size=args.per_device_eval_batch_size
    )

    model = EdgeSimMPNN(in_feats=384, hid_feats=384, out_feats=384, edge_feats=384, n_layers=5, dropout=0.1)

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

    model = model.to(device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    temp = 0.05
    sim_fct = Similarity(temp)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0
    eval_loss = 0
    loss_fct = nn.CrossEntropyLoss()
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, (nodes1, nodes2, m1, m2, edge_m1, edge_m2, bm1, bm2) in enumerate(train_dataloader):
            nodes1, m1, edge_m1, bm1 = nodes1.to(device), m1.to(device), edge_m1.to(device), bm1.to(device)
            nodes2, m2, edge_m2, bm2 = nodes2.to(device), m2.to(device), edge_m2.to(device), bm2.to(device)

            output1 = model(x=nodes1, m=m1, em=edge_m1, bm=bm1)
            anchor = output1

            output2 = model(x=nodes2, m=m2, em=edge_m2, bm=bm2)
            pos = output2

            # output3 = model(x=nodes3, m=m3, em=edge_m3, bm=bm3)
            # neg = output3

            # good_sim_score = F.cosine_similarity(anchor, pos)
            # bad_sim_score = F.cosine_similarity(anchor, neg)
            # ce_output = torch.cat([good_sim_score, bad_sim_score])
            # labels = torch.tensor([1] * args.per_device_train_batch_size + [0] * args.per_device_train_batch_size, dtype=torch.float32).to()

            z1_z2_cos = sim_fct(anchor.unsqueeze(1), pos.unsqueeze(0))
            labels = torch.arange(z1_z2_cos.size(0)).long().to(device)
            loss = loss_fct(z1_z2_cos, labels)
            
            # loss = triplet_loss(anchor, pos, neg)
            # loss = loss_fct(ce_output, labels)

            optimizer.zero_grad()

            total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            if args.with_tracking:
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

        losses = []  
        mrr_list = []      # mrr list
        sim_pair_cos = []  # debug use, save sim pair cos similarity
        model.eval()
        for step, (nodes1, nodes2, m1, m2, edge_m1, edge_m2, bm1, bm2) in enumerate(eval_dataloader):
            with torch.no_grad():
                nodes1, m1, edge_m1, bm1 = nodes1.to(device), m1.to(device), edge_m1.to(device), bm1.to(device)
                nodes2, m2, edge_m2, bm2 = nodes2.to(device), m2.to(device), edge_m2.to(device), bm2.to(device)
                # nodes3, m3, edge_m3, bm3 = nodes3.to(device), m3.to(device), edge_m3.to(device), bm3.to(device)

                output1 = model(x=nodes1, m=m1, em=edge_m1, bm=bm1)
                anchor = output1

                output2 = model(x=nodes2, m=m2, em=edge_m2, bm=bm2)
                pos = output2

                # output3 = model(x=nodes3, m=m3, em=edge_m3, bm=bm3)
                # neg = output3

            rank_reciprocal_sum = 0          # rank_reciprocal_sum: save sum of maximum sim func rank reciprocal in this batch
            mrr = 0                          # mrr: sum of sim function rank reciprocal and calculate avg
            for i in range(len(anchor)): 
                vA = anchor[i: i+1].cpu()    # vA: target func embedding 
                sim = []                     # sim:save all sim between target func and other func
                for j in range(len(pos)):
                    vB = pos[j: j+1].cpu()   # vB: other func embedding, when i == j means vA should sim vB
                    AB_sim = F.cosine_similarity(vA, vB).item()                        
                    sim.append(AB_sim)
                sim_list = np.array(sim)
                sim_rank_idx_list = np.argsort(-sim_list)
                vA_correct_rank = 0
                for j in range(len(pos)):
                    if sim_rank_idx_list[j] == i:                            
                        vA_correct_rank = j + 1

                sim_pair_cos.append(sim[i])

                rank_reciprocal_sum += 1 / vA_correct_rank

            mrr = rank_reciprocal_sum / len(anchor)
            mrr_list.append(mrr)
            
            # good_sim_score = F.cosine_similarity(anchor, pos)
            # bad_sim_score = F.cosine_similarity(anchor, neg)
            # ce_output = torch.cat([good_sim_score, bad_sim_score])
            # labels = torch.tensor([1] * args.per_device_train_batch_size + [0] * args.per_device_train_batch_size, dtype=torch.float32)
            
            z1_z2_cos = sim_fct(anchor.unsqueeze(1), pos.unsqueeze(0))
            labels = torch.arange(z1_z2_cos.size(0)).long().to(device)
            loss = loss_fct(z1_z2_cos, labels)

            # loss = triplet_loss(anchor, pos, neg)
            # loss = loss_fct(ce_output, labels)
            losses.append(loss)
        losses = torch.stack(losses)
        try:
            eval_loss = torch.mean(losses)
            avg_mrr = np.mean(np.array(mrr_list))
        except OverflowError:
            print(f'[!]eval overflow')

        logger.info(f"epoch {epoch}: eval_loss:{eval_loss}: mrr: {avg_mrr}")

        if args.with_tracking:
            wandb.log(
                {
                    "eval_loss": eval_loss,
                    "avg_train_loss": total_loss.item() / len(train_dataloader),
                    "avg_mrr": avg_mrr,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, f"finetune_epoch_{epoch+1}"))
            

    if args.output_dir is not None:
        torch.save(model.cpu().state_dict(), os.path.join(args.output_dir, f"finetune_epoch_{epoch+1}"))


if __name__ == "__main__":
    main()
