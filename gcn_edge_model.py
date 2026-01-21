import torch
import torch.nn as nn
import math


def gcn_maxpooling(x, bm):
    batch_size = torch.max(bm) + 1
    out = []
    for i in range(batch_size):
        inds = (bm == i).nonzero()[:, 0]
        x_ind = torch.index_select(x, dim=0, index=inds)
        out.append(torch.max(x_ind, dim=0, keepdim=False)[0])
    out = torch.stack(out, dim=0)

    return out

def gcn_meanpooling(x, bm):
    batch_size = torch.max(bm) + 1
    out = []
    for i in range(batch_size):
        inds = (bm == i).nonzero()[:, 0]
        x_ind = torch.index_select(x, dim=0, index=inds)
        out.append(torch.mean(x_ind, dim=0, keepdim=False))
    out = torch.stack(out, dim=0)

    return out

def gcn_sumpooling(x, bm):
    batch_size = torch.max(bm) + 1
    out = []
    for i in range(batch_size):
        inds = (bm == i).nonzero()[:, 0]
        x_ind = torch.index_select(x, dim=0, index=inds)
        out.append(torch.sum(x_ind, dim=0, keepdim=False))
    out = torch.stack(out, dim=0)

    return out


class EdgeGCN_DIR_CAT(nn.Module):
    def __init__(self, vec_dim, out_dim, edge_dim, dropout=0., use_bias=True):
        super(EdgeGCN_DIR_CAT, self).__init__()
        self.in_features = vec_dim
        self.out_features = out_dim

        self.weight_x = nn.Parameter(torch.FloatTensor(vec_dim, int(out_dim/2)))

        self.weight_node_in = nn.Parameter(torch.FloatTensor(vec_dim, int(out_dim/2)))
        self.weight_node_out = nn.Parameter(torch.FloatTensor(vec_dim, int(out_dim/2)))
        self.weight_edge_in = nn.Parameter(torch.FloatTensor(edge_dim, int(out_dim)))
        self.weight_edge_out = nn.Parameter(torch.FloatTensor(edge_dim, int(out_dim)))
        self.weight_aggressive = nn.Parameter(torch.FloatTensor(out_dim*3, out_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.nonlinear = nn.ReLU()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_node_in.size(1))
        self.weight_node_in.data.uniform_(-stdv, stdv)
        self.weight_node_out.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_edge_in.size(1))
        self.weight_edge_in.data.uniform_(-stdv, stdv)
        self.weight_edge_out.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_aggressive.size(1))
        self.weight_aggressive.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor, edge_in_feat_matrix: torch.Tensor, edge_out_feat_matrix: torch.Tensor):
        support_in = torch.mm(x, self.weight_node_in)
        support_out = torch.mm(x, self.weight_node_out)
        node_in_output = torch.spmm(adj_matrix.T, support_in)
        node_out_output = torch.spmm(adj_matrix, support_out)

        edge_in_reshaped = edge_in_feat_matrix.view(-1, edge_in_feat_matrix.size(-1))
        edge_in_output_h = torch.matmul(edge_in_reshaped, self.weight_edge_in)
        edge_in_output_m = edge_in_output_h.view(*edge_in_feat_matrix.shape[:-1], self.weight_edge_in.size(-1))
        edge_in_output = edge_in_output_m.sum(dim=1)

        edge_out_reshaped = edge_out_feat_matrix.view(-1, edge_out_feat_matrix.size(-1))
        edge_out_output_h = torch.matmul(edge_out_reshaped, self.weight_edge_out)
        edge_out_output_m = edge_out_output_h.view(*edge_out_feat_matrix.shape[:-1], self.weight_edge_out.size(-1))
        edge_out_output = edge_out_output_m.sum(dim=0)

        cat_node_edge = torch.cat((node_in_output, node_out_output, edge_in_output, edge_out_output), dim=1)

        output = torch.mm(cat_node_edge, self.weight_aggressive)

        if self.bias is not None:
            output = output + self.bias
        
        output = self.nonlinear(output)
        output = self.dropout(output)

        return output, edge_in_output_m, edge_out_output_m

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class EdgeSimMPNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, edge_feats, n_layers, dropout=0.):
        super(EdgeSimMPNN, self).__init__()

        self.n_layers = n_layers
        gcns = []
        for _ in range(n_layers):
            gcns.append(EdgeGCN_DIR_CAT(in_feats, hid_feats, edge_feats, dropout, use_bias=True))
        self.gcns = nn.ModuleList(gcns)
        # self.trans = nn.Linear(in_feats, hid_feats)
        # self.gcn = EdgeGCN_TypeD_MY_DIR_CAT(in_feats, hid_feats, edge_feats, dropout, use_bias=True)
        self.update = nn.GRU(input_size=hid_feats, hidden_size=in_feats, dropout=0.3)
        self.MLP = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )

    def forward(self, x, m, em, bm):
        # hid = self.trans(x)
        mp, e_in, e_out = self.gcns[0](x, m, em, em)
        _, hid = self.update(mp.unsqueeze(0), x.unsqueeze(0))
        hid = torch.squeeze(hid)
        for l_n in range(self.n_layers - 1):
            mp, e_in, e_out = self.gcns[l_n+1](hid, m, e_in, e_out)
            _, hid = self.update(mp.unsqueeze(0), hid.unsqueeze(0))
            hid = torch.squeeze(hid)
        
        h_0 = self.MLP(x)
        h = self.MLP(hid)

        hid = h_0 + h

        output = gcn_maxpooling(hid, bm)  # (batch_size, out_dim)

        return output

