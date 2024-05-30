import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from utils.search_space import act_map
from torch_geometric.nn import LayerNorm
from utils.operations import *


class NaOp(nn.Module):
    def __init__(self, primitive, in_dim, out_dim, act, with_linear=False):
        super(NaOp, self).__init__()

        self._op = NA_OPS[primitive](in_dim, out_dim)
        self.op_linear = nn.Linear(in_dim, out_dim)
        self.act = act_map(act)
        self.with_linear = with_linear

    def forward(self, x, edge_index):
        if self.with_linear:
            return self.act(self._op(x, edge_index) + self.op_linear(x))
        else:
            return self.act(self._op(x, edge_index))

class ScOp(nn.Module):
    def __init__(self, primitive):
        super(ScOp, self).__init__()
        self._op = SC_OPS[primitive]()

    def forward(self, x):
        return self._op(x)


class LaOp(nn.Module):
    def __init__(self, primitive, hidden_size, act, num_layers=None):
        super(LaOp, self).__init__()
        self._op = FF_OPS[primitive](hidden_size, num_layers)
        self.act = act_map(act)

    def forward(self, x):
        return self.act(self._op(x))


class NodeClassificationOutputModule(nn.Module):
    def __init__(self, node_embedding_dim, num_classes):
        super(NodeClassificationOutputModule, self).__init__()
        self.linear1 = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.linear2 = nn.Linear(node_embedding_dim, num_classes)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class GraphClassificationOutputModule(nn.Module):
    def __init__(self, node_embedding_dim, hidden_dim, num_classes):
        super(GraphClassificationOutputModule, self).__init__()
        self.linear1 = nn.Linear(node_embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, batch):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.linear2(x)
        return x


class LinkPredictionOutputModule(nn.Module):
    def __init__(self, node_embedding_dim):
        super(LinkPredictionOutputModule, self).__init__()
        self.linear_a = nn.Linear(node_embedding_dim, node_embedding_dim)
        # self.linear_b = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.linear = nn.Linear(2 * node_embedding_dim, 1)

    def forward(self, inputs, pos_edge_index, neg_edge_index):
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        node_a = torch.index_select(inputs, 0, total_edge_index[0])  # 按行取多行正
        node_a = self.linear_a(node_a)
        node_b = torch.index_select(inputs, 0, total_edge_index[1])  # 按行取多行负
        node_b = self.linear_a(node_b)
        x = torch.cat((node_a, node_b), 1)
        x = self.linear(x)
        x = torch.clamp(torch.sigmoid(x), min=1e-8, max=1 - 1e-8)
        return x


class MTLAGLNet(torch.nn.Module):

    def __init__(self, tasks, actions, in_dim, output_gc_dim, output_nc_dim, layer_nums, drop_outs, args):
        super(MTLAGLNet, self).__init__()

        self.name = "MTLAGLNet"

        # args
        self.tasks = tasks
        self.actions = actions  # (layer_nums) * ('na_primitives'),  ((layer_nums + 2) * (layer_nums + 1) / 2) * (
        # 'sc_primitives'), (layer_nums+ 1) * ('la_primitives'), 1 * 'references'
        self.in_dim = in_dim
        self.output_gc_dim = output_gc_dim
        self.output_nc_dim = output_nc_dim
        self.layer_nums = layer_nums
        self.hidden_size = args.embedding_dim
        self.activation_function = args.act
        self.dropout = drop_outs
        self.args = args
        self.fix_last = True  # fix last layer in design architectures
        self.num_edge = int((self.layer_nums + 2) * (self.layer_nums + 1) / 2)
        self.num_param_tasks_model = {"gc": 4, "nc": 4, "lp": 4}

        if self.args.search_agg:
            self.na_primitives = actions[:self.layer_nums]
            self.sc_primitives = actions[self.layer_nums:(self.layer_nums + self.num_edge)]
            self.la_primitives = actions[(self.layer_nums + self.num_edge):(2*self.layer_nums + self.num_edge + 1)]
            self.reference = actions[-1]
        else:
            self.na_primitives = [self.args.agg] * self.layer_nums
            self.sc_primitives = actions[:self.num_edge]
            self.la_primitives = actions[self.num_edge:(self.num_edge + self.layer_nums + 1)]
            self.reference = actions[-1]

        # pre-process
        self.lin1 = nn.Linear(self.in_dim, self.hidden_size)

        # node aggregator op
        self.gnn_layers = nn.ModuleList(
            [NaOp(self.na_primitives[i], self.hidden_size, self.hidden_size, self.activation_function, with_linear=args.with_linear) for i in
             range(self.layer_nums)])

        self.bns = torch.nn.ModuleList()
        if self.args.batch_norm:
            for i in range(self.layer_nums):
                self.bns.append(torch.nn.BatchNorm1d(self.hidden_size))

        self.lns = torch.nn.ModuleList()
        if self.args.layer_norm:
            for i in range(self.layer_nums):
                self.lns.append(LayerNorm(self.hidden_size, affine=True))

        # selection
        self.skip_op = nn.ModuleList()
        for i in range(self.num_edge):
            self.skip_op.append(ScOp(self.sc_primitives[i]))

        # layer aggregator op (fuse function)
        self.fuse_funcs = nn.ModuleList()
        for i in range(self.layer_nums + 1):
            self.fuse_funcs.append(LaOp(self.la_primitives[i], self.hidden_size, 'linear', num_layers=i + 1))


        if "gc" in self.tasks:
            self.gc_output_layer = GraphClassificationOutputModule(self.hidden_size, self.hidden_size,
                                                                   self.output_gc_dim)
        if "nc" in self.tasks:
            self.nc_output_layer = NodeClassificationOutputModule(self.hidden_size, self.output_nc_dim)
        if "lp" in self.tasks:
            self.lp_output_layer = LinkPredictionOutputModule(self.hidden_size)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # input node
        features = []
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        features += [x]

        # intermediate nodes
        start = 0
        for layer in range(self.layer_nums):

            # select inputs
            layer_input = []
            for i in range(layer + 1):
                edge_id = start + i
                layer_input += [self.skip_op[edge_id](features[i])]

            # fuse features
            tmp_input = self.fuse_funcs[layer](layer_input)

            # aggegation
            x = self.gnn_layers[layer](tmp_input, edge_index)
            if self.args.batch_norm:
                x = self.bns[layer](x)
            if self.args.layer_norm:
                x = self.lns[layer](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # output
            features += [x]
            start += (layer + 1)

        # select features for output node
        output_features = []
        for i in range(self.layer_nums + 1):
            edge_id = start + i
            output_features += [self.skip_op[edge_id](features[i])]
        output = self.fuse_funcs[-1](output_features)
        output = F.dropout(output, p=self.dropout, training=self.training)

        gc_output = self.gc_output_layer(output, data.batch) if "gc" in self.tasks else None
        nc_output = self.nc_output_layer(output) if "nc" in self.tasks else None
        lp_output = self.lp_output_layer(output, data.pos_edge_index, data.neg_edge_index) if "lp" in self.tasks else None
        return gc_output, nc_output, lp_output