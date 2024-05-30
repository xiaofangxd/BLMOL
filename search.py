import numpy as np
import torch
import configs
import sys
import os
import geatpy as ea
from utils.NSGAII import moea_NSGA2_templet
from utils.NAS_Problem import GraphNAS

def main(args):

    problem = GraphNAS(args)

    # def outFunc(alg, pop):
    #     print('The %d gen' % alg.currentGen)

    if args.algUL == 'NSGAII':
        algorithm = moea_NSGA2_templet(
            problem,
            ea.Population(Encoding='RI', NIND=args.num_individuals),
            MAXGEN=args.num_generations,  # 最大进化代数
            logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
            # outFunc=outFunc)


    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=False,
                      drawLog=False,
                      saveFlag=False)
    u, indices = np.unique(res['ObjV'], axis=0, return_index=True)
    data = res['Vars'][indices, :]
    fitness = []
    for i in range(len(data)):
        net = problem.decode(list(data[i, :]))
        # run gnn to get the classification accuracy as fitness
        model, model_train_acc, model_val_acc, model_test_acc, infer_time = problem.gnn_manager.train(net,
                                                                                                   problem.param_genes)
        fitness.append(np.array(list(model_val_acc.values())))

    print(fitness)


if __name__ == "__main__":

    args = configs.build_args('BLMOL')
    args.gpu = 1
    args.device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
    torch.manual_seed(args.random_seed)
    args.dataset = 'ENZYMES'
    args.tasks = ["gc", "nc"]
    args.algLL = 'LS'
    args.algUL = 'NSGAII' # NAGAII
    args.n_test_rays = 20
    # args.save = "surrogate_model_data/" + args.dataset
    args.save = "surrogate_model_data/" + args.dataset + '_' + args.algLL
    args.search_log = "search_log/" + args.dataset + '_' + args.algLL
    if not os.path.exists(args.save):
        os.makedirs(args.save,exist_ok=True)
    if not os.path.exists(args.search_log):
        os.makedirs(args.search_log,exist_ok=True)
    args.num_individuals = 100
    args.num_generations = 200
    args.epochs = 500
    args.batch_size = 256
    args.search_agg = False
    args.agg = 'gcn'
    args.act = 'relu'
    args.embedding_dim = 256
    args.num_gnn_layers = 4
    # args.layer_norm = True
    # args.batch_norm = True
    # args.with_linear = True
    main(args)
