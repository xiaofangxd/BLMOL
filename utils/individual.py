import os
import pickle
from utils.search_space import net_space
import numpy as np


class Individual(object):

    def __init__(self, args, net_genes, param_genes):
        self.net_genes_encode = None
        self.model_train_acc = None
        self.model_val_acc = None
        self.model_test_acc = None
        self.infer_time = None
        self.fitness = None
        self.args = args
        self.net_genes = net_genes
        self.param_genes = param_genes
        self.encode()

    def get_net_genes(self):
        return self.net_genes

    def get_param_genes(self):
        return self.param_genes

    def cal_fitness(self, gnn_manager):
        # run gnn to get the classification accuracy as fitness
        model, model_train_acc, model_val_acc, model_test_acc, infer_time = gnn_manager.train(self.net_genes,
                                                                                              self.param_genes)
        self.model_train_acc = model_train_acc
        self.model_val_acc = model_val_acc
        self.model_test_acc = model_test_acc
        self.infer_time = infer_time
        self.fitness = np.array(list(model_val_acc.values()))

    def cal_fitness_by_surrogate_model(self, surrogate_model):
        # run surrogate_model to get the classification accuracy as fitness
        fitness = []
        for i in range(len(surrogate_model)):
            fitness.append(surrogate_model[i]['trained_model'].predict(np.array(self.net_genes_encode).reshape(1, -1)))
        self.fitness = np.array(fitness)[:, 0]

    def cal_target(self, gnn_manager, index_str):
        # run gnn to get the classification accuracy as fitness
        model, model_train_acc, model_val_acc, model_test_acc, infer_time = gnn_manager.train_data(self.net_genes,
                                                                                                   self.param_genes,
                                                                                                   index_str)
        self.model_train_acc = model_train_acc
        self.model_val_acc = model_val_acc
        self.model_test_acc = model_test_acc
        self.infer_time = infer_time
        self.fitness = np.array(list(model_val_acc.values()))

    def save_surrogate_model(self, index_str):
        """ Saves the surrogate model. """
        # net_genes (list), parameter_genes (list), model_train_acc (dict), model_val_acc (dict), model_test_acc (
        # dict), infer_time (list)
        filename = os.path.join(self.args.save + '/' + index_str)
        # save net_genes and parameter_genes
        result = {'net_architecture': self.net_genes, 'net_hype_parameter': self.param_genes,
                  'model_train_acc': self.model_train_acc, 'model_val_acc': self.model_val_acc,
                  'model_test_acc': self.model_test_acc, 'infer_time': self.infer_time}

        with open(filename + '.pkl', "wb") as fp:  # Pickling
            pickle.dump(result, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open(filename + '.txt', 'w') as f:
            f.write("net_architecture:\n")
            for i in self.net_genes:
                if isinstance(i, str):
                    f.write(i)
                else:
                    f.write(str(i))
                f.write("\t")
            f.write("\nnet_hype-parameter:\n")
            for i in self.param_genes:
                f.write(f"{i:>7.7f}")
                f.write("\t")
            f.write("\nmodel_train_acc:\n")
            for i in self.model_train_acc:
                f.write(i + ":{}\n".format(self.model_train_acc[i]))
                f.write("\t")
            f.write("\nmodel_val_acc:\n")
            for i in self.model_val_acc:
                f.write(i + ":{}\n".format(self.model_val_acc[i]))
                f.write("\t")
            f.write("\nmodel_test_acc:\n")
            for i in self.model_test_acc:
                f.write(i + ":{}\n".format(self.model_test_acc[i]))
                f.write("\t")
            f.write("\nmodel_infer_time(train, val, test):\n")
            for i in self.infer_time:
                f.write(str(i))
                f.write("\t")

    def get_fitness(self):
        return self.fitness

    def get_test_acc(self):
        return self.test_acc

    def mutation_net_gene(self, mutate_point, new_gene, type='struct'):
        if type == 'struct':
            self.net_genes[mutate_point] = new_gene
        else:
            raise Exception("wrong mutation type")

    def encode(self):
        encode = []
        b = ''
        for i in self.args.tasks:
            b = b + i
        reference = np.load(self.args.save + '/' + b + '/reference.npy')
        num_edge = int((self.args.num_gnn_layers + 2) * (self.args.num_gnn_layers + 1) / 2)
        if self.args.search_agg:
            ag_parm = self.net_genes[:self.args.num_gnn_layers]
            ag_parm_T = [net_space['na_primitives'].index(ag_parm[ind]) for ind in range(len(ag_parm))]
            encode.extend(ag_parm_T)

            sc_parm = self.net_genes[self.args.num_gnn_layers:(self.args.num_gnn_layers + num_edge)]
            sc_parm_T = [net_space['sc_primitives'].index(sc_parm[ind]) for ind in range(len(sc_parm))]
            encode.extend(sc_parm_T)

            la_parm = self.net_genes[
                      (self.args.num_gnn_layers + num_edge):(2 * self.args.num_gnn_layers + num_edge + 1)]
            la_parm_T = [net_space['la_primitives'].index(la_parm[ind]) for ind in range(len(la_parm))]
            encode.extend(la_parm_T)

            ref_parm = np.where(reference == self.net_genes[-1])[0][0]
            encode.append(ref_parm)
        else:
            sc_parm = self.net_genes[:num_edge]
            sc_parm_T = [net_space['sc_primitives'].index(sc_parm[ind]) for ind in range(len(sc_parm))]
            encode.extend(sc_parm_T)

            la_parm = self.net_genes[num_edge:(num_edge + self.args.num_gnn_layers + 1)]
            la_parm_T = [net_space['la_primitives'].index(la_parm[ind]) for ind in range(len(la_parm))]
            encode.extend(la_parm_T)

            ref_parm = np.where(reference == self.net_genes[-1])[0][0]
            encode.append(ref_parm)
        self.net_genes_encode = encode
