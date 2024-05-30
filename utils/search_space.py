import torch
from random import sample
from utils.MOO_utils import circle_points
import numpy as np
import os
# net_space = {
#             'na_primitives': ['sage', 'sage_sum', 'sage_max', 'gcn', 'gin', 'gat', 'gat_sym', 'gat_cos', 'gat_linear', 'gat_generalized_linear', 'geniepath'],
#             'activation_function': ['sigmoid', 'tanh', 'relu', 'linear',
#                                    'softplus', 'leaky_relu', 'relu6', 'elu'],
#             'sc_primitives': ['none', 'skip'],
#             'la_primitives': ['l_max', 'l_concat', 'l_lstm'],
#             'hidden_units': [32, 64, 128, 256, 512]
#             }

# param_space = {
#     'drop_out': [0.05, 0.2, 0.4, 0.6],
#     'learning_rate': [5e-4, 1e-3, 5e-3, 1e-2],
#     'weight_decay': [5e-4, 8e-4, 1e-3, 4e-3],
#     'alpha': [0.1, 0.7, 1.2],
#     'lamda': [0.01, 0.1, 1, 3, 5]
# }


net_space = {
    'na_primitives': ['sage', 'gcn', 'gin', 'gat'],
    'sc_primitives': ['zero', 'identity'],
    'la_primitives': ['max', 'concat', 'mean', 'sum', 'lstm', 'att'],
}

param_space = {
    'drop_out': [0.5],
    'learning_rate': [1e-3],
    'weight_decay': [1e-4],
}

class HybridSearchSpace(object):
    def __init__(self, args):

        self.args = args
        self.num_gnn_layers = args.num_gnn_layers
        self.tasks = args.tasks
        self.test_rays = circle_points(args.n_test_rays, dim=len(self.tasks))
        b = ''
        for i in args.tasks:
            b = b + i
        if not os.path.exists(args.save + '/' + b):
            os.mkdir(args.save + '/' + b)
        np.save(os.path.join(args.save + '/' + b, 'reference'), self.test_rays)

        self.num_edges = int((self.num_gnn_layers + 2) * (self.num_gnn_layers + 1) / 2)
        
        self.net_space = net_space
        self.net_space['references'] = list(self.test_rays)
        # {
        #     'na_primitives': ['sage', 'gcn', 'gin', 'gat'],
        #     'sc_primitives': ['zero', 'identity'],
        #     'la_primitives': ['max', 'concat', 'mean', 'sum', 'lstm', 'att'],
        #     'references': list(self.test_rays)
        #     }
            
        self.param_space = param_space

    def get_net_space(self):
        return self.net_space
    
    def get_param_space(self):
        return self.param_space   
    
    def get_action_type_list(self):
        action_names = []
        if self.args.search_agg:
            action_names.append(['na_primitives'] * self.num_gnn_layers)
        action_names.extend(['sc_primitives'] * self.num_edges)
        action_names.extend(['la_primitives'] * (self.num_gnn_layers + 1))
        action_names.append('references')
        return action_names
    
    def get_net_instance(self):
        "sample network architects for multi-layer GNN"
        net_architects = []
        net_space = self.get_net_space()
        if self.args.search_agg:
            for i in range(self.num_gnn_layers):
                actions = net_space['na_primitives']
                net_architects.extend(sample(actions, 1))
        for i in range(self.num_edges):
            actions = net_space['sc_primitives']
            net_architects.extend(sample(actions, 1))
        for i in range(self.num_gnn_layers + 1):
            actions = net_space['la_primitives']
            net_architects.extend(sample(actions, 1))
        actions = net_space['references']
        net_architects.extend(sample(actions, 1))
            
        return net_architects
    
    def get_param_instance(self):
        "sample network hyper parameters"
        net_parameters = []
        param_space = self.get_param_space()
        params = param_space['drop_out']
        net_parameters.extend(sample(params, 1))
        params = param_space['learning_rate']
        net_parameters.extend(sample(params, 1))
        params = param_space['weight_decay']
        net_parameters.extend(sample(params, 1))

        return net_parameters
        
        
    def get_one_net_gene(self):
        "randomly sample a gene for mutation from the architecture space"
        action_type_list = self.get_action_type_list()
        gene_mutate_index = sample(range(len(action_type_list)), 1)[0]
        gene_mutate_candidates = self.net_space[action_type_list[gene_mutate_index]]
        gene_mutate_to = sample(gene_mutate_candidates, 1)[0]
        
        return gene_mutate_index, gene_mutate_to
    
    def get_one_param_gene(self):
        "randomly sample a gene for mutation from the param space"
        param_len = len(self.param_space)
        param_type_list = list(self.param_space.keys())
        gene_mutate_index = sample(range(param_len), 1)[0]
        gene_mutate_candidates = self.param_space[param_type_list[gene_mutate_index]]
        gene_mutate_to = sample(gene_mutate_candidates, 1)[0]
        
        return gene_mutate_index, gene_mutate_to
            
        
        
        
        
def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")
        