from utils.search_space import HybridSearchSpace
import os
import numpy as np
import geatpy as ea
import pickle
from utils.gnn_model_manager import GNNModelManager

class GraphNAS(ea.Problem):
    def __init__(self, args):
        name = 'GraphNAS'
        self.args = args
        self.search_space = HybridSearchSpace(args)
        self.net_genes = self.search_space.get_action_type_list()
        self.net_space = self.search_space.get_net_space()
        self.Dim = len(self.net_genes)
        self.M = len(args.tasks) # number of objective
        self.maxormins = [-1] * self.M # max or min objective，1: min; -1: max
        self.varTypes = [1] * self.Dim  # type of x，0: real number, 1: integer
        self.lb = [0] * self.Dim # lower bound
        self.ub = [len(self.net_space[i]) for i in self.net_genes] # upper bound
        self.lbin = [1] * self.Dim  # lower boundary（0: nor including，1: including）
        self.ubin = [0] * self.Dim  # upper boundary
        self.surrogate_model = None
        self.gnn_manager = None
        self.param_genes = self.search_space.get_param_instance()
        # prepare data set for training the gnn model
        self.load_training_data()

        self.load_surrogate_model()
        ea.Problem.__init__(self, name, self.M, self.maxormins, self.Dim, self.varTypes, self.lb, self.ub, self.lbin, self.ubin)

    def load_training_data(self):
        self.gnn_manager = GNNModelManager(self.args)
        self.gnn_manager.load_data()

        # # dataset statistics
        # print(self.gnn_manager.data)

    def load_surrogate_model(self):
        surrogate_model = []
        b = ''
        for i in self.args.tasks:
            b = b + i
        for i in range(len(self.args.tasks)):
            filename = os.path.join(self.args.save + '/' + b + '/surr_data_' + str(i))
            with open(filename + '.pkl', "rb") as fp:
                surrogate_model.append(pickle.load(fp))
        self.surrogate_model = surrogate_model

    def decode(self, individual):
        net = []
        for ind, gene in enumerate(individual):
            net.append(self.net_space[self.net_genes[ind]][gene])
        return net

    def aimFunc(self, pop):  
        Vars = pop.Phen  
        fitness = []
        for i in range(len(self.surrogate_model)):
            fitness.append(self.surrogate_model[i]['trained_model'].predict(Vars))
        pop.ObjV = np.array(fitness).transpose()

    def aimFunc_real(self, pop):  
        Vars = pop.Phen  
        fitness = []
        for i in range(len(Vars)):
            net = self.decode(list(Vars[i, :]))
            # run gnn to get the classification accuracy as fitness
            model, model_train_acc, model_val_acc, model_test_acc, infer_time = self.gnn_manager.train(net, self.param_genes)
            fitness.append(np.array(list(model_val_acc.values())))
        pop.ObjV = np.array(fitness)

    def evaluation(self, pop, call_real=False):

        """
        Description:
            Call aimFunc() or evalVars() to calculate the objective function value and constraint violation degree of the passed population.
        Note:
        If the subclass overrides both aimFunc and evalVars, evalVars will be invalid.

        Input parameters:
            pop : class <Population> - Population object.

        """

        aimFuncCallFlag = False
        evalVarsCallFlag = False
       
        if self.aimFunc.__name__ != 'wrapper': 
            if self.aimFunc.__module__ != 'geatpy.Problem':
                aimFuncCallFlag = True
        else: 
            aimFuncCallFlag = True
        if aimFuncCallFlag:
            if call_real:
                self.aimFunc_real(pop)
            else:
                self.aimFunc(pop)
        else: 
            if self.evalVars.__name__ != 'wrapper':
                if self.evalVars.__module__ != 'geatpy.Problem':
                    evalVarsCallFlag = True
            else:  
                evalVarsCallFlag = True
        if evalVarsCallFlag:
            return_object = self.evalVars(pop.Phen)
            if type(return_object) != tuple:
                pop.ObjV = return_object
            else:
                pop.ObjV, pop.CV = return_object
        if not aimFuncCallFlag and not evalVarsCallFlag:
            raise RuntimeError('error in Problem: one of the function aimFunc and evalVars should be rewritten.)')