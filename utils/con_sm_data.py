import os
import pickle
import numpy as np
from utils.search_space import net_space
from scipy.spatial.distance import squareform
from torch_geometric.data import Data
import torch
from torch_geometric.loader import DataLoader


def con_graph_dataset(data, args):
    data_list = []
    for i in range(len(data)):
        x = torch.tensor(data[i]['individual']['feature_node'], dtype=torch.float)
        edge_index = torch.tensor(data[i]['individual']['edge_index'], dtype=torch.long)
        y = torch.tensor(np.array(list(data[i]['model_val_acc'].values())), dtype=torch.float)
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        graph_data.graph_feature = torch.tensor(data[i]['individual']['feature_graph'], dtype=torch.float)
        data_list.append(graph_data)
    loader = DataLoader(data_list, batch_size=args.batch_size_SM, shuffle=True)
    return loader


def prep_data(data, args):
    b = ''
    for i in args.tasks:
        b = b + i
    reference = np.load(args.save + '/' + b + '/reference.npy')
    num_edge = int((args.num_gnn_layers + 2) * (args.num_gnn_layers + 1) / 2)
    for i in range(args.num_individuals):
        actions = data[i]['net_architecture']
        encode = []
        y = np.array(list(data[i]['model_val_acc'].values()))
        if args.search_agg:
            ag_parm = actions[:args.num_gnn_layers]
            ag_parm_T = [net_space['na_primitives'].index(ag_parm[ind]) for ind in range(len(ag_parm))]
            encode.extend(ag_parm_T)

            sc_parm = actions[args.num_gnn_layers:(args.num_gnn_layers + num_edge)]
            sc_parm_T = [net_space['sc_primitives'].index(sc_parm[ind]) for ind in range(len(sc_parm))]
            encode.extend(sc_parm_T)

            sc_array_T = np.triu(squareform(sc_parm_T))
            tmp = np.where(sc_array_T == 1)
            edge_index = np.dstack((tmp[0], tmp[1])).squeeze()

            la_parm = actions[(args.num_gnn_layers + num_edge):(2 * args.num_gnn_layers + num_edge + 1)]
            la_parm_T = [net_space['la_primitives'].index(la_parm[ind]) for ind in range(len(la_parm))]
            encode.extend(la_parm_T)

            ref_parm = np.where(reference == actions[-1])[0][0]
            encode.append(ref_parm)

            feat_dim_ag = len(net_space['na_primitives']) + 2
            ag_parm_T.insert(0, feat_dim_ag - 2)  # inputs
            ag_parm_T.append(feat_dim_ag - 1)  # outputs

            feat_dim_la = len(net_space['la_primitives']) + 1
            la_parm_T.insert(0, feat_dim_la - 1)  # input node
            feature_node = np.concatenate((np.eye(feat_dim_ag)[ag_parm_T], np.eye(feat_dim_la)[la_parm_T]), axis=1)

            feat_ref_dim = len(reference)
            feature_graph = np.eye(feat_ref_dim)[ref_parm]

            individual = {'encode': np.array(encode), 'ag_parm_T': ag_parm_T, 'sc_parm_T': sc_parm_T, 'sc_array_T': sc_array_T,
                          'edge_index': edge_index, 'la_parm_T': la_parm_T, 'feature_node': feature_node,
                          'ref_parm': ref_parm, 'feature_graph': feature_graph, 'y': y}
        else:
            sc_parm = actions[:num_edge]
            sc_parm_T = [net_space['sc_primitives'].index(sc_parm[ind]) for ind in range(len(sc_parm))]
            encode.extend(sc_parm_T)

            sc_array_T = np.triu(squareform(sc_parm_T))
            tmp = np.where(sc_array_T == 1)
            edge_index = np.dstack((tmp[0], tmp[1])).squeeze()

            la_parm = actions[num_edge:(num_edge + args.num_gnn_layers + 1)]
            la_parm_T = [net_space['la_primitives'].index(la_parm[ind]) for ind in range(len(la_parm))]
            encode.extend(la_parm_T)

            ref_parm = np.where(reference == actions[-1])[0][0]
            encode.append(ref_parm)

            feat_dim = len(net_space['la_primitives']) + 1
            la_parm_T.insert(0, feat_dim - 1)  # input node
            feature_node = np.eye(feat_dim)[la_parm_T]

            feat_ref_dim = len(reference)
            feature_graph = np.eye(feat_ref_dim)[ref_parm]

            individual = {'encode': np.array(encode), 'sc_parm_T': sc_parm_T, 'sc_array_T': sc_array_T, 'edge_index': edge_index,
                          'la_parm_T': la_parm_T, 'feature_node': feature_node, 'ref_parm': ref_parm,
                          'feature_graph': feature_graph, 'y': y}
        data[i]['individual'] = individual
    return data


def load_data(args):
    data = []
    index_str = ""
    for j in args.tasks:
        index_str += j + '_'
    for i in range(args.num_individuals):
        index_strs = index_str + '{}'.format(i)
        filename = os.path.join(args.save + '/' + index_strs + '.pkl')
        with open(filename, "rb") as fp:  # Pickling
            mydict = pickle.load(fp)
        data.append(mydict)
    data = prep_data(data, args)
    return data

def pre_ML_data(dataset):
    X = dataset[0]['individual']['encode']
    Y = dataset[0]['individual']['y']
    for ind in range(len(dataset)):
        if ind != 0:
            X = np.vstack((X, dataset[ind]['individual']['encode']))
            Y = np.vstack((Y, dataset[ind]['individual']['y']))
    return X, Y


