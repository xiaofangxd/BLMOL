from utils.search_space import HybridSearchSpace
from utils.individual import Individual
from utils.gnn_model_manager import GNNModelManager


class Surrogate(object):

    def __init__(self, args):

        self.data = None
        self.gnn_manager = None
        self.args = args
        hybrid_search_space = HybridSearchSpace(self.args)
        self.hybrid_search_space = hybrid_search_space
        

    def load_training_data(self):
        self.gnn_manager = GNNModelManager(self.args)
        self.gnn_manager.load_data()

        # dataset statistics
        print(self.gnn_manager.data)

    def init_data(self):

        data = []

        for i in range(self.args.num_individuals):
            net_genes = self.hybrid_search_space.get_net_instance()
            param_genes = self.hybrid_search_space.get_param_instance()
            instance = Individual(self.args, net_genes, param_genes)
            data.append(instance)

        self.data = data

    def cal_target(self):
        """calculate fitness scores of all individuals,
          e.g., the classification accuracy from GNN"""
        index_str = ""
        for j in self.args.tasks:
            index_str += j + '_'

        for index, individual in enumerate(self.data):
            index_strs = index_str + '{}'.format(index)
            individual.cal_target(self.gnn_manager, index_strs)
            individual.save_surrogate_model(index_strs)

    def construct_surrogate_model_data(self):
        # prepare data set for training the gnn model
        self.load_training_data()

        # prepare data of struct and parameter
        self.init_data()

        # calculate target
        self.cal_target()