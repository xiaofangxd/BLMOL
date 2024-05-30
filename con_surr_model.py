import numpy as np
import torch
import os
import configs
from utils.con_sm_data import load_data, con_graph_dataset, pre_ML_data
from utils.surrogate_model import try_different_method, model, method
from sklearn.model_selection import train_test_split
import pickle

def main(args):
    data = load_data(args)
    graph_dataset = con_graph_dataset(data, args)
    print(graph_dataset)
    sm_data_X, sm_data_Y = pre_ML_data(data)
    x_train, x_test, y_train, y_test = train_test_split(sm_data_X, sm_data_Y, test_size=0.1, random_state=22)
    print('----------------------train---------------------')
    # 'decision_tree', 'linear_regression', 'svm', 'knn', 'random_forest', 'adaboost', 'GBRT', 'Bagging',
    #      'ExtraTree', 'Gaussian_Process', 'MLP'

    for j in range(len(args.tasks)):
        trained_model = []
        KTau = []
        MSE = []
        for i in range(0, 11):
            trained_modell, KTauu, MSEE = try_different_method(x_train, y_train[:, j], x_test, y_test[:, j], model[i], method[i], show_fig=False, return_flag=True)
            trained_model.append(trained_modell)
            KTau.append(KTauu)
            MSE.append(MSEE)

        index = np.argsort(KTau)
        surr_data = {}
        surr_data['name_id'] = index[-1]
        surr_data['KTau'] = KTau[index[-1]]
        surr_data['MSE'] = MSE[index[-1]]
        surr_data['trained_model'] = trained_model[index[-1]]
        b = ''
        for i in args.tasks:
            b = b + i
        filename = os.path.join(args.save + '/' + b + '/surr_data_' + str(j))
        if not os.path.exists(args.save + '/' + b):
            os.mkdir(args.save + '/' + b)
        with open(filename + '.pkl', "wb") as fp:  # Pickling
            pickle.dump(surr_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(filename + '.pkl', "rb") as fp:
        #     test = pickle.load(fp)
        # print(test)
    return surr_data



if __name__ == "__main__":
    args = configs.build_args('BLMOL')
    args.dataset = 'ENZYMES'
    args.tasks = ["gc", "nc"]
    args.algLL = 'LS'
    args.save = "surrogate_model_data/" + args.dataset + '_' + args.algLL
    # args.save = "surrogate_model_data/" + args.dataset
    args.gpu = 1
    args.device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    args.num_individuals = 10
    args.search_agg = False
    args.num_gnn_layers = 4
    args.batch_size_SM = 128
    if not os.path.exists(args.save):
        print("Error!")
    surr_data = main(args)
