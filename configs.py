# -*- coding: utf-8 -*-

import argparse

def build_args(model_name):
    
    parser = argparse.ArgumentParser(description=model_name)
    register_default_args(parser)
    args = parser.parse_args()

    return args

def register_default_args(parser): 
    
    # general settings
    parser.add_argument('--random_seed', type=int, default=123)    
    parser.add_argument("--dataset", type=str, default="ENZYMES",
                        help="Name of the dataset from the TUDortmund.")
    parser.add_argument("--data-folder", type=str, default="data",
                        help="Path to the folder where data will be stored (default is data/).")
    parser.add_argument('--save', type=str, default='result', help='experiment result')
    parser.add_argument('--search_log', type=str, default='search_log', help='experiment log')
    # parser.add_argument('--plot', type=bool, default=False, help='whether to plot PF')
    parser.add_argument("--tasks", type=list, default=["gc", "nc", "lp"],
                        help="Tasks to be performed (default is 'gc,nc,lp').")
    parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
    parser.add_argument("--n_test_rays", type=int, default=20,
                        help="Number of test preference vectors for Pareto front generating methods.")
    parser.add_argument("--reference_point", type=list, default=[2, 2, 2],
                        help="Reference point for hyper-volume calculation.")
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument("--algLL", type=str, default="LS",
                        help="MOO algorithm of lower-layer sub-problem.(LS)")
    parser.add_argument("--algUL", type=str, default="NSGAII",
                        help="MOO algorithm of upper-layer sub-problem. (NSGAII)")

    # settings for the genetic algorithm
    parser.add_argument('--num_individuals', type=int, default=20,
                        help='the population size')
    parser.add_argument('--num_generations', type=int, default=100,
                        help='number of evolving generations')

    # settings for the gnn model
    parser.add_argument('--num_gnn_layers', type=int, default=3,
                        help='number of the GNN layers')
    parser.add_argument('--with_linear', type=bool, default=False, help=' in NAMixOp with linear op')
    parser.add_argument('--layer_norm', action='store_true', default=False, help='whether to use layer norm')

    parser.add_argument('--search_agg', type=bool, default=True, help=' search agg')
    parser.add_argument('--agg', type=str, default='sage', help='if not search agg, aggregations used in this framework')

    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Number of tasks in a mini-batch of tasks (default: 256).")
    parser.add_argument("--batch-size-SM", type=int, default=32,
                        help="batch-size of surrogate model (default: 32).")
    parser.add_argument("--embedding-dim", type=int, default=32,
                        help="Node embedding dimension (default: 32).")
    parser.add_argument('--act', type=str, default='relu', help='activation_function, relu, elu')
    parser.add_argument('--batch_norm', type=bool, default=False, help='use batch norm in trainging supernet.')
    
    
    
               