import os
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import math
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    DataCollatorForLanguageModeling
)

MAX_STEP = 50

class FinetuneDatasetCE(Dataset):
    def __init__(self, graph_list, model, device, tokenizer, max_seq_length, is_tensor=False):
        super(FinetuneDatasetCE, self).__init__()
        self.graphs = graph_list
        self.model = model
        self.device =device
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # mapping all raw grap data into embedding form
        self.is_tensor = is_tensor
        if is_tensor:
            self.data_list = graph_list
        else:
            self.data_list = self.map_data(graph_list)
            self.model.cpu()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        cfg1 = self.data_list[index][0]
        cfg2 = self.data_list[index][1]

        if self.is_tensor:
            return (self.list_to_tensor(cfg1[0]), cfg1[1], self.list_to_tensor(cfg1[2])), (self.list_to_tensor(cfg2[0]), cfg2[1], self.list_to_tensor(cfg2[2]))

        return (cfg1['node_feat'], cfg1['edge_pairs'], cfg1['edge_feat']), (cfg2['node_feat'], cfg2['edge_pairs'], cfg2['edge_feat'])
    
    # traverse all data and transfer to embeddings
    def map_data(self, raw_graph_list):
        res_list = []
        progress_bar = tqdm(range(len(raw_graph_list)))
        for data_tri_tuple in raw_graph_list:
            target_func = data_tri_tuple[0]
            sim_func = data_tri_tuple[1]

            target_nodes = target_func['nodes']
            target_preds = [target_nodes[edge[0]] for edge in target_func['edges']]
            target_succs = [target_nodes[edge[1]] for edge in target_func['edges']]
            nodes1 = self.gen_block_edge_feat_plus(target_nodes, target_func['arch'])
            edges1 = self.gen_block_edge_feat_plus(target_preds, target_func['arch'], is_pair=True, batch2=target_succs)

            sim_nodes = sim_func['nodes']
            sim_preds = [sim_nodes[edge[0]] for edge in sim_func['edges']]
            sim_succs = [sim_nodes[edge[1]] for edge in sim_func['edges']]
            nodes2 = self.gen_block_edge_feat_plus(sim_nodes, sim_func['arch'])
            edges2 = self.gen_block_edge_feat_plus(sim_preds, sim_func['arch'], is_pair=True, batch2=sim_succs)

            res_list.append([
                {
                    'node_feat': nodes1,
                    'edge_pairs': target_func['edges'],
                    'edge_feat': edges1,
                },
                {
                    'node_feat': nodes2,
                    'edge_pairs': sim_func['edges'],
                    'edge_feat': edges2,
                },
                ])
            progress_bar.update(1)
        
        return res_list

    def gen_block_edge_feat_plus(self, batch1, arch, is_pair=False, batch2=None):
        if len(batch1) == 0:
            return torch.tensor([])

        if is_pair:
            result_input = self.tokenizer(
                batch1,
                batch2,            
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_special_tokens_mask=False,
            )
        else:
            result_input = self.tokenizer(
                batch1,        
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_special_tokens_mask=False,
            )
        result_input['arch_ids'] = [[arch]*self.max_seq_length for _ in batch1]

        step = MAX_STEP
        batch_len = len(batch1)
        if batch_len <= step:
            return self.gen_embedding(**result_input)
        iter = math.ceil(batch_len / step)
        partition = []
        
        for i in range(iter):
            temp_feat = self.gen_embedding(result_input['input_ids'][i * step: min((i + 1) * step, batch_len)],
                                           result_input['attention_mask'][i * step: min((i + 1) * step, batch_len)],
                                           result_input['token_type_ids'][i * step: min((i + 1) * step, batch_len)],
                                           result_input['arch_ids'][i * step: min((i + 1) * step, batch_len)])
            partition.append(temp_feat)
        return torch.cat(partition, dim=0)
        
    def gen_embedding(self, input_ids, attention_mask, token_type_ids, arch_ids):
        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_mask = torch.LongTensor(attention_mask).to(self.device)
        token_type_ids = torch.LongTensor(token_type_ids).to(self.device)
        arch_ids = torch.LongTensor(arch_ids).to(self.device)

        self.model.to(self.device)
        self.model.eval()

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, arch_ids=arch_ids)
        pooled_output = output[0][:, 0].detach().cpu()
        return pooled_output

    def list_to_tensor(self, t_list):
        return torch.Tensor(t_list)


# Dataset for finetune gnn without using edge feat 
class FinetuneNoEdgeDataset(Dataset):
    def __init__(self, graph_list, model, device, tokenizer, max_seq_length, is_tensor=False):
        super(FinetuneNoEdgeDataset, self).__init__()
        self.graphs = graph_list
        self.model = model
        self.device =device
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # mapping all raw grap data into embedding form
        self.is_tensor = is_tensor
        if is_tensor:
            self.data_list = graph_list
        else:
            self.data_list = self.map_data(graph_list)
            self.model.cpu()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        cfg1 = self.data_list[index][0]
        cfg2 = self.data_list[index][1]

        if self.is_tensor:
            return (self.list_to_tensor(cfg1[0]), cfg1[1], self.list_to_tensor(cfg1[2])), (self.list_to_tensor(cfg2[0]), cfg2[1], self.list_to_tensor(cfg2[2]))

        return (cfg1['node_feat'], cfg1['edge_pairs']), (cfg2['node_feat'], cfg2['edge_pairs'])
    
    # traverse all data and transfer to embeddings
    def map_data(self, raw_graph_list):
        res_list = []
        progress_bar = tqdm(range(len(raw_graph_list)))
        for data_tri_tuple in raw_graph_list:
            target_func = data_tri_tuple[0]
            sim_func = data_tri_tuple[1]
            unsim_func = data_tri_tuple[2]

            target_nodes = target_func['nodes']
            nodes1 = self.gen_block_feat(target_nodes, target_func['arch'])
            
            sim_nodes = sim_func['nodes']
            nodes2 = self.gen_block_feat(sim_nodes, sim_func['arch'])

            unsim_nodes = unsim_func['nodes']
            nodes3 = self.gen_block_feat(unsim_nodes, unsim_func['arch'])

            res_list.append([
                {
                    'node_feat': nodes1,
                    'edge_pairs': target_func['edges'],
                },
                {
                    'node_feat': nodes2,
                    'edge_pairs': sim_func['edges'],
                },
                {
                    'node_feat': nodes3,
                    'edge_pairs': unsim_func['edges'],
                }])
            progress_bar.update(1)
        
        return res_list

    def gen_block_feat(self, batch1, arch):
        if len(batch1) == 0:
            return torch.tensor([])

        result_input = self.tokenizer(
            batch1,        
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_special_tokens_mask=False,
        )
        result_input['arch_ids'] = [[arch]*self.max_seq_length for _ in batch1]

        step = 32
        batch_len = len(batch1)
        if batch_len <= step:
            return self.gen_embedding(**result_input)
        iter = math.ceil(batch_len / step)
        partition = []
        
        for i in tqdm(range(iter)):
            temp_feat = self.gen_embedding(result_input['input_ids'][i * step: min((i + 1) * step, batch_len)],
                                           result_input['attention_mask'][i * step: min((i + 1) * step, batch_len)],
                                           result_input['token_type_ids'][i * step: min((i + 1) * step, batch_len)],
                                           result_input['arch_ids'][i * step: min((i + 1) * step, batch_len)])
            partition.append(temp_feat)
        return torch.cat(partition, dim=0)
        
    def gen_embedding(self, input_ids, attention_mask, token_type_ids, arch_ids):
        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_mask = torch.LongTensor(attention_mask).to(self.device)
        token_type_ids = torch.LongTensor(token_type_ids).to(self.device)
        arch_ids = torch.LongTensor(arch_ids).to(self.device)

        self.model.to(self.device)
        self.model.eval()

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, arch_ids=arch_ids)
        pooled_output = output[0][:, 0].detach().cpu()
        return pooled_output

    def list_to_tensor(self, t_list):
        return torch.Tensor(t_list)