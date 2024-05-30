import torch
import configs
import sys
from utils.construct_surrogate_model_data import Surrogate
import os
import pickle

def main(args):
    args.device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
    torch.manual_seed(args.random_seed)

    args.dataset = 'ENZYMES'
    args.tasks = ["gc", "nc"]
    args.algLL = 'LS'
    args.n_test_rays = 20
    # args.save = "surrogate_model_data/" + args.dataset
    args.save = "surrogate_model_data/" + args.dataset + '_' + args.algLL
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    args.num_individuals = 100
    args.epochs = 200
    args.batch_size = 256
    args.search_agg = False
    args.agg = 'gcn'
    args.act = 'relu'
    args.embedding_dim = 256
    args.num_gnn_layers = 4
    # args.layer_norm = True
    # args.batch_norm = True
    # args.with_linear = True
    surrogate = Surrogate(args)
    surrogate.construct_surrogate_model_data()


if __name__ == "__main__":

    args = configs.build_args('BLMOL')
    main(args)
    argsDict = args.__dict__

    filename = os.path.join(args.save + '/args.pkl')
    with open(filename, "wb") as fp:  # Pickling
        pickle.dump(argsDict, fp, protocol=pickle.HIGHEST_PROTOCOL)
