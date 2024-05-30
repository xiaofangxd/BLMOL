import os
import copy
import pickle
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# different methods
# 1.decision tree regression
model_decision_tree_regression = tree.DecisionTreeRegressor()

# 2.linear regression
model_linear_regression = LinearRegression()

# 3.SVM regression
model_svm = svm.SVR()

# 4.kNN regression
model_k_neighbor = neighbors.KNeighborsRegressor()

# 5.random forest regression
model_random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=230)

# 6.Adaboost regression
model_adaboost_regressor = ensemble.AdaBoostRegressor()

# 7.GBRT regression
model_gradient_boosting_regressor = ensemble.GradientBoostingRegressor()

# 8.Bagging regression
model_bagging_regressor = ensemble.BaggingRegressor()

# 9.ExtraTree regression
model_extra_tree_regressor = ExtraTreeRegressor()

# 10.Gaussian Process Regression
model_gaussian_process_regressor = GaussianProcessRegressor()

# 11.MLP Regression
model_MLP_regressor = MLPRegressor()

model = [model_decision_tree_regression, model_linear_regression, model_svm, model_k_neighbor,
         model_random_forest_regressor, model_adaboost_regressor, model_gradient_boosting_regressor,
         model_bagging_regressor, model_extra_tree_regressor, model_gaussian_process_regressor, model_MLP_regressor]

method = ['decision_tree', 'linear_regression', 'svm', 'knn', 'random_forest', 'adaboost', 'GBRT', 'Bagging',
          'ExtraTree', 'Gaussian_Process', 'MLP']

def Random_forestAndmore_test(method, x_train, y_train, x_test, y_test, more_x_test, more_y_test, parameter_scale,
                              step=10):
    print('First run without more test data')
    no_more_test_KTau_list, no_more_test_MSE_list = Ablation_study(method, x_train, y_train, x_test, y_test,
                                                                   parameter_scale, step, more_test_data=False,
                                                                   show_figure=False)

    print('Second run with more test data')
    more_test_KTau_list, more_test_MSE_list = Ablation_study(method, x_train, y_train, more_x_test, more_y_test,
                                                             parameter_scale, step, more_test_data=True,
                                                             show_figure=False)

    # save KTau and MSE list
    if not os.path.isdir('pkl'):
        os.makedirs('pkl')
    save_path = r'pkl\ktau_and_mse_list.pkl'
    save_dic = {'no_more_test_KTau_list': no_more_test_KTau_list, 'no_more_test_MSE_list': no_more_test_MSE_list,
                'more_test_KTau_list': more_test_KTau_list, 'more_test_MSE_list': more_test_MSE_list}
    with open(save_path, 'wb') as file:
        pickle.dump(save_dic, file)
        print('Save KTau and MSE list successfully!')

    # Because running the above code is a time-consuming process. To speed up the process,
    # we recommend using the function make_plot_for_KTau_and_MSE with the saving data.
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), no_more_test_KTau_list[4:], "hotpink",
             linestyle=':', marker='^', label="original test")
    plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), more_test_KTau_list[4:], "skyblue",
             linestyle=':', marker='D', label="committee prediction")
    plt.xlabel("number of estimators")
    plt.ylabel("KTau")
    plt.legend(loc="best")
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), no_more_test_MSE_list[4:], "hotpink",
             linestyle=':', marker='^', label="original test")
    plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), more_test_MSE_list[4:], "skyblue",
             linestyle=':', marker='D', label="committee prediction")
    plt.xlabel("number of estimators")
    plt.ylabel("MSE")
    plt.legend(loc="best")
    plt.show()

def Ablation_study(method, x_train, y_train, x_test, y_test, parameter_scale, step=1, more_test_data=False,
                   show_figure=True, upper_tri_num=120):
    score_list, KTau_list, MSE_list = [], [], []
    y_test_copy = copy.deepcopy(y_test)
    print(f'Ablation study, method: {method}')
    for parameter in range(parameter_scale[0], parameter_scale[-1], step):
        if method == 'random_forest':
            Model = ensemble.RandomForestRegressor(n_estimators=parameter)
        elif method == 'knn':
            Model = neighbors.KNeighborsRegressor(n_neighbors=parameter)
        elif method == 'GBRT':
            Model = ensemble.GradientBoostingRegressor(n_estimators=parameter)
        elif method == 'Bagging':
            Model = ensemble.BaggingRegressor(n_estimators=parameter)
        else:
            raise ValueError
        Model.fit(x_train, y_train)
        result = Model.predict(x_test)
        if more_test_data:
            if isinstance(upper_tri_num, list):
                print('Select upper tri matrix!!!')
                mean_result = []
                first_ground_truth = []
                last_num = 0
                for num in upper_tri_num:
                    mean_result.extend([np.mean(result[last_num:last_num + num])])
                    first_ground_truth.extend([y_test_copy[last_num]])
                    last_num = last_num + num
            else:
                # the number is fixed 120 now, use the mean value
                # that is one metric can produce 5! = 120 same architecture representation
                mean_result = []
                first_ground_truth = []
                for i in range(len(x_test) // 120):
                    mean_result.extend([np.mean(result[i * 120:(i + 1) * 120])])
                    first_ground_truth.extend([y_test_copy[i * 120]])
                    # because the ground truth in [i*120:(i+1)*120] are all [i*120]
                result = mean_result
                y_test = first_ground_truth
        result = list(result)
        score = r2_score(y_test, result)
        result_arg = np.argsort(result)
        y_test_arg = np.argsort(y_test)
        result_rank = np.zeros(len(y_test_arg))
        y_test_rank = np.zeros(len(y_test_arg))
        for i in range(len(y_test_arg)):
            result_rank[result_arg[i]] = i
            y_test_rank[y_test_arg[i]] = i
        KTau, _ = kendalltau(result_rank, y_test_rank)
        MSE = calculate_MSE(y_test, result)
        print(
            'parameter: {:}, KTau: {:}, MSE: {:}, R2score: {:}'.format(parameter, KTau, MSE, score))
        score_list.append([score])
        MSE_list.append([MSE])
        KTau_list.append([KTau])

    # save KTau and MSE list
    if not os.path.isdir('pkl'):
        os.makedirs('pkl')
    save_path = r'pkl\ktau_and_mse_list.pkl'
    save_dic = {'KTau_list': KTau_list, 'MSE_list': MSE_list}
    with open(save_path, 'wb') as file:
        pickle.dump(save_dic, file)
        print('Save KTau and MSE list successfully!')

    if show_figure:
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), KTau_list, "hotpink", linestyle=':',
                 marker='^')
        plt.xlabel("number of estimators")
        plt.ylabel("KTau")

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), MSE_list, "skyblue", linestyle=':',
                 marker='D')
        plt.xlabel("number of estimators")
        plt.ylabel("MSE")

        plt.title(f"Different number of estimators in {method}")
        plt.show()

def make_plot_for_KTau_and_MSE(KTau_list, MSE_list):
    # show the results of estimators = {50, 60, ..., 300}
    parameter_scale = [10, 310]
    # don't change the step
    step = 10
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), KTau_list, "hotpink", linestyle=':',
             marker='^')
    plt.xlabel("number of estimators")
    plt.ylabel("KTau")

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(parameter_scale[0], parameter_scale[-1], step), MSE_list, "skyblue", linestyle=':',
             marker='D')
    plt.xlabel("number of estimators")
    plt.ylabel("MSE")
    plt.show()


def calculate_MSE(x, y):
    # input two list, x: predict, y: ground truth
    # output MSE
    mse_list = np.array([(element_x - element_y) ** 2 for element_x, element_y in zip(x, y)])
    mse = np.mean(mse_list)
    return mse

def try_different_method(x_train, y_train, x_test, y_test, model, method, show_fig=True, return_flag=False):
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    result = list(result)
    score = r2_score(y_test, result)
    result_arg = np.argsort(result)
    y_test_arg = np.argsort(y_test)
    result_rank = np.zeros(len(y_test_arg))
    y_test_rank = np.zeros(len(y_test_arg))
    for i in range(len(y_test_arg)):
        result_rank[result_arg[i]] = i
        y_test_rank[y_test_arg[i]] = i
    KTau, _ = kendalltau(result_rank, y_test_rank)
    print('method: {:}, KTau: {:}, MSE: {:}, R2score: {:}'.format(method, KTau, calculate_MSE(y_test, result), score))
    print('--------------------try-end---------------------\n')
    if show_fig:
        x = np.arange(0, 1, 0.01)
        y = x
        plt.figure(figsize=(5, 5))
        plt.plot(x, y, 'g', label='y_test = result')
        plt.scatter(result, y_test, s=1)
        plt.xlabel("predict_result")
        plt.ylabel("y_test")
        plt.title(f"method:{method}---score:{score}")
        plt.legend(loc="best")
        plt.show()

        x = np.arange(0, len(y_test), 0.1)
        y = x
        plt.figure(figsize=(6, 6))
        line_color = '#1F77D0'
        plt.plot(x, y, c=line_color, linewidth=1)
        point_color = '#FF4400'
        plt.scatter(result_rank, y_test_rank, c=point_color, s=2)
        plt.xlabel("predict_result")
        plt.ylabel("y_test")
        plt.title(f"method:{method}---KTau:{KTau}")
        plt.xlim(xmax=5000, xmin=0)
        plt.ylim(ymax=5000, ymin=0)
        plt.show()

    if return_flag:
        return model, KTau, calculate_MSE(y_test, result)


