import os
import time
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils.gnn import MTLAGLNet
from utils.dataload_util import *
from tqdm import tqdm
import warnings
import utils.bl_utils as bl_utils
from torch.autograd import Variable
from utils.epo_lp import EPO_LP
import utils.MOO_utils as MOO_utils

warnings.filterwarnings('ignore')

def evaluate(model, dataloader, args):
    model.eval()
    epoch_stats = bl_utils.EpochStats()
    infer_time = 0
    for batch_idx, batch in enumerate(dataloader):
        _, test_batch, _ = multi_task_train_test_split(batch, False, tasks=args.tasks)
        test_batch = test_batch[0]
        test_batch = test_batch.to(args.device)
        with torch.no_grad():
            begin_time = time.time()
            gc_test_logit, nc_test_logit, lp_test_logit = model(test_batch)
            endtime = time.time() - begin_time
            infer_time += endtime
            # GC
            if "gc" in args.tasks:
                gc_loss = F.cross_entropy(gc_test_logit, test_batch.y)
                with torch.no_grad():
                    gc_acc = MOO_utils.get_accuracy(gc_test_logit, test_batch.y)
                epoch_stats.update("gc", test_batch, gc_loss, gc_acc, False)
            # NC
            if "nc" in args.tasks:
                node_labels = test_batch.node_y.argmax(1)
                train_mask = test_batch.train_mask.squeeze()
                test_mask = (train_mask == 0).float()
                nc_loss = F.cross_entropy(nc_test_logit[test_mask == 1], node_labels[test_mask == 1])
                with torch.no_grad():
                    nc_acc = MOO_utils.get_accuracy(nc_test_logit[test_mask == 1], node_labels[test_mask == 1])
                epoch_stats.update("nc", test_batch, nc_loss, nc_acc, False)
            # LP
            if "lp" in args.tasks:
                test_link_labels = get_link_labels(test_batch.pos_edge_index, test_batch.neg_edge_index)
                lp_loss = F.binary_cross_entropy_with_logits(lp_test_logit.squeeze(), test_link_labels)
                with torch.no_grad():
                    test_labels = test_link_labels.detach().cpu().numpy()
                    test_predictions = lp_test_logit.detach().cpu().numpy()
                    lp_acc = roc_auc_score(test_labels, test_predictions.squeeze())
                epoch_stats.update("lp", test_batch, lp_loss, lp_acc, False)

    infer_time = infer_time/len(dataloader)
    tasks_test_stats = epoch_stats.get_average_stats()
    test_acc_sum, test_acc = bl_utils.task_accs_and_losses(tasks_test_stats)

    return test_acc_sum, test_acc, infer_time

class GNNModelManager(object):

    def __init__(self, args):

        self.args = args
        self.loss_fn = [torch.nn.functional.cross_entropy,
                        torch.nn.functional.cross_entropy,
                        torch.nn.functional.binary_cross_entropy_with_logits]

    def load_data(self):
        dataset, train_val_test_ratio = get_graph_dataset(self.args.dataset, destination_dir=self.args.data_folder)
        dataset = dataset.shuffle()
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(dataset, self.args.batch_size, "multi",
                                                                            train_val_test_ratio=[0.7, 0.1, 0.2],
                                                                            num_workers=4, shuffle_train=True)
        data = [train_dataloader, val_dataloader, test_dataloader]
        output_gc_dim = dataset.num_classes
        output_nc_dim = dataset[0].node_y.size(1)
        num_node_features = dataset.num_node_features

        self.data = data
        self.args.output_gc_dim = output_gc_dim
        self.args.output_nc_dim = output_nc_dim
        self.args.num_node_features = num_node_features

    def build_gnn(self, actions, drop_outs):
        model = MTLAGLNet(self.args.tasks, actions, self.args.num_node_features, self.args.output_gc_dim,
                          self.args.output_nc_dim, self.args.num_gnn_layers, drop_outs, self.args)
        return model

    def train_data(self, actions=None, params=None, index_str=""):
        print('==================================\ncurrent training actions={}, params={}'.format(actions, params))

        # create gnn model
        drop_outs = params[0]
        learning_rate = params[1]
        weight_decay = params[2]
        reference = actions[-1]

        gnn_model = self.build_gnn(actions, drop_outs)
        gnn_model.to(device=self.args.device)
        # define optimizer
        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)

        # run model to get accuracy
        model, model_train_acc, model_val_acc, model_test_acc, infer_time = self.run_model_data(gnn_model, optimizer, self.loss_fn, self.data, self.args.epochs,
                                                  reference, show_info=True, index_str=index_str)

        return model, model_train_acc, model_val_acc, model_test_acc, infer_time

    def train(self, actions=None, params=None):
        print('==================================\ncurrent training actions={}, params={}'.format(actions, params))

        # create gnn model
        drop_outs = params[0]
        learning_rate = params[1]
        weight_decay = params[2]
        reference = actions[-1]

        gnn_model = self.build_gnn(actions, drop_outs)
        gnn_model.to(device=self.args.device)
        # define optimizer
        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)

        # run model to get accuracy
        model, model_train_acc, model_val_acc, model_test_acc, infer_time = self.run_model(gnn_model, optimizer, self.loss_fn, self.data, self.args.epochs,
                                                  reference, show_info=True)

        return model, model_train_acc, model_val_acc, model_test_acc, infer_time

    def run_model_data(self, model, optimizer, loss_fn, data, epochs, reference, return_best=False,
                  show_info=False, index_str=""):

        begin_time = time.time()
        best_per = 0
        best_performance = {task: 0.0 for task in self.args.tasks}
        best_val_acc_all = 0
        model_train_acc = {task: 0.0 for task in self.args.tasks}
        model_val_acc = {task: 0.0 for task in self.args.tasks}
        model_test_acc = {task: 0.0 for task in self.args.tasks}

        _, n_params = MOO_utils.getNumParams(model.parameters())
        print(f"# params={n_params}")
        epo_lp = EPO_LP(m=len(self.args.tasks), n=n_params, r=reference)
        n_manual_adjusts = 0
        descent = 0.

        #         print("Number of train datas:", data.train_mask.sum())
        for epoch in range(epochs):
            # train
            model.train()
            # for batch_idx, batch in enumerate(tqdm(data[0], desc="Train_Batch")):
            for batch_idx, batch in enumerate(data[0]):
                _, train_batch, _ = multi_task_train_test_split(batch, True, tasks=self.args.tasks)
                train_batch = train_batch[0]
                train_batch = train_batch.to(self.args.device)
                alpha = reference / reference.sum()

                alpha = len(self.args.tasks) * torch.from_numpy(alpha).to(self.args.device)
                print(alpha)
                weight = {}
                for i, t in enumerate(self.args.tasks):
                    weight[t] = alpha[i]

                # optimization step
                losses = {}
                optimizer.zero_grad()
                # Forward pass
                gc_train_logit, nc_train_logit, lp_train_logit = model(train_batch)

                if "gc" in self.args.tasks:
                    losses["gc"] = loss_fn[0](gc_train_logit, train_batch.y)
                if "nc" in self.args.tasks:
                    node_labels = train_batch.node_y.argmax(1)
                    train_mask = train_batch.train_mask.squeeze()
                    losses["nc"] = loss_fn[1](nc_train_logit[train_mask == 1], node_labels[train_mask == 1])
                if "lp" in self.args.tasks:
                    train_link_labels = get_link_labels(train_batch.pos_edge_index, train_batch.neg_edge_index)
                    losses["lp"] = loss_fn[2](lp_train_logit.squeeze(), train_link_labels)
                loss = 0
                # for t in self.args.tasks:
                #    loss = loss + losses[t]

                for t in self.args.tasks:
                    loss = loss + weight[t] * losses[t]

                loss.backward()
                optimizer.step()

            # print(f"\tdescent={descent / len(data[0])}")
            # if n_manual_adjusts > 0:
            #     print(f"\t # manual tweek={n_manual_adjusts}")

            # evaluate
            train_acc_sum, train_acc, train_infer_time = evaluate(model, data[0], self.args)
            val_acc_sum, val_acc, val_infer_time = evaluate(model, data[1], self.args)
            test_acc_sum, test_acc, test_infer_time = evaluate(model, data[2], self.args)

            if val_acc_sum > best_val_acc_all:  # and train_loss < min_train_loss
                best_val_acc_all = val_acc_sum
                model_train_acc = train_acc
                model_val_acc = val_acc
                model_test_acc = test_acc
                infer_time = [train_infer_time, val_infer_time, test_infer_time]
                MOO_utils.save(model, os.path.join(self.args.save, index_str + '_weights.pt'))
                if test_acc_sum > best_per:
                    best_per = test_acc_sum
                    best_performance = test_acc
            if show_info:
                time_used = time.time() - begin_time
                print(
                    "Epoch {:05d} | loss {:.4f} train_acc {} | val_acc_sum {:.4f} | test_acc_sum {:.4f} | time {}".format(
                        epoch, loss, train_acc, val_acc_sum, test_acc_sum, time_used))

        if return_best:
            return model, model_train_acc, model_val_acc, best_performance, infer_time
        else:
            return model, model_train_acc, model_val_acc, model_test_acc, infer_time

    def run_model(self, model, optimizer, loss_fn, data, epochs, reference, return_best=False,
                  show_info=False):

        begin_time = time.time()
        best_per = 0
        best_performance = {task: 0.0 for task in self.args.tasks}
        best_val_acc_all = 0
        model_train_acc = {task: 0.0 for task in self.args.tasks}
        model_val_acc = {task: 0.0 for task in self.args.tasks}
        model_test_acc = {task: 0.0 for task in self.args.tasks}

        _, n_params = MOO_utils.getNumParams(model.parameters())
        print(f"# params={n_params}")
        epo_lp = EPO_LP(m=len(self.args.tasks), n=n_params, r=reference)
        n_manual_adjusts = 0
        descent = 0.

        #         print("Number of train datas:", data.train_mask.sum())
        for epoch in range(epochs):
            # train
            model.train()
            # for batch_idx, batch in enumerate(tqdm(data[0], desc="Train_Batch")):
            for batch_idx, batch in enumerate(data[0]):
                _, train_batch, _ = multi_task_train_test_split(batch, True, tasks=self.args.tasks)
                train_batch = train_batch[0]
                train_batch = train_batch.to(self.args.device)
                alpha = reference / reference.sum()

                alpha = len(self.args.tasks) * torch.from_numpy(alpha).to(self.args.device)
                print(alpha)
                weight = {}
                for i, t in enumerate(self.args.tasks):
                    weight[t] = alpha[i]

                # optimization step
                losses = {}
                optimizer.zero_grad()
                # Forward pass
                gc_train_logit, nc_train_logit, lp_train_logit = model(train_batch)

                if "gc" in self.args.tasks:
                    losses["gc"] = loss_fn[0](gc_train_logit, train_batch.y)
                if "nc" in self.args.tasks:
                    node_labels = train_batch.node_y.argmax(1)
                    train_mask = train_batch.train_mask.squeeze()
                    losses["nc"] = loss_fn[1](nc_train_logit[train_mask == 1], node_labels[train_mask == 1])
                if "lp" in self.args.tasks:
                    train_link_labels = get_link_labels(train_batch.pos_edge_index, train_batch.neg_edge_index)
                    losses["lp"] = loss_fn[2](lp_train_logit.squeeze(), train_link_labels)
                loss = 0
                # for t in self.args.tasks:
                #    loss = loss + losses[t]

                for t in self.args.tasks:
                    loss = loss + weight[t] * losses[t]

                loss.backward()
                optimizer.step()

            # print(f"\tdescent={descent / len(data[0])}")
            # if n_manual_adjusts > 0:
            #     print(f"\t # manual tweek={n_manual_adjusts}")

            # evaluate
            train_acc_sum, train_acc, train_infer_time = evaluate(model, data[0], self.args)
            val_acc_sum, val_acc, val_infer_time = evaluate(model, data[1], self.args)
            test_acc_sum, test_acc, test_infer_time = evaluate(model, data[2], self.args)

            if val_acc_sum > best_val_acc_all:  # and train_loss < min_train_loss
                best_val_acc_all = val_acc_sum
                model_train_acc = train_acc
                model_val_acc = val_acc
                model_test_acc = test_acc
                infer_time = [train_infer_time, val_infer_time, test_infer_time]
                if test_acc_sum > best_per:
                    best_per = test_acc_sum
                    best_performance = test_acc
            if show_info:
                time_used = time.time() - begin_time
                print(
                    "Epoch {:05d} | loss {:.4f} train_acc {} | val_acc_sum {:.4f} | test_acc_sum {:.4f} | time {}".format(
                        epoch, loss, train_acc, val_acc_sum, test_acc_sum, time_used))

        if return_best:
            return model, model_train_acc, model_val_acc, best_performance, infer_time
        else:
            return model, model_train_acc, model_val_acc, model_test_acc, infer_time