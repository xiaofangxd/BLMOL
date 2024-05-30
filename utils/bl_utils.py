import numpy as np
from tqdm import tqdm


class EpochStats:
    def __init__(self, args=None):
        self.tasks = ["gc", "nc", "lp"]
        self.task_correct = {task: [] for task in self.tasks}
        self.task_losses = {task: [] for task in self.tasks}
        self.num_gc_graphs = 0
        self.num_nc_nodes = 0
        self.num_lp_batches = 0
        self.num_lp_edges = 0

        if args != None:
            self.n_test_rays = args.n_test_rays
            self.task_correct_all = {task: {i: [] for i in range(args.n_test_rays)} for task in self.tasks}
            self.task_losses_all = {task: {i: [] for i in range(args.n_test_rays)} for task in self.tasks}

    def update(self, task, batch, loss, acc, train):
        if task == "gc":
            self.num_gc_graphs += batch.num_graphs
            self.task_losses[task].append((loss * batch.num_graphs).item())
            self.task_correct[task].append((acc * batch.num_graphs).item())
        elif task == "nc":
            num_train_nodes = batch.train_mask.sum().item()
            num_test_nodes = batch.batch.size(0) - num_train_nodes
            if train:
                self.num_nc_nodes += num_train_nodes
                self.task_losses[task].append((loss * num_train_nodes).item())
                self.task_correct[task].append((acc * num_train_nodes).item())
            else:
                self.num_nc_nodes += num_test_nodes
                self.task_losses[task].append((loss * num_test_nodes).item())
                self.task_correct[task].append((acc * num_test_nodes).item())
        elif task == "lp":
            self.num_lp_batches += 1
            num_lp_edges = batch.neg_edge_index.size(1) + batch.pos_edge_index.size(1)
            self.num_lp_edges += num_lp_edges
            self.task_losses[task].append((loss * num_lp_edges).item())
            self.task_correct[task].append(acc.item())

    def updatenum(self, task, batch, train):
        if task == "gc":
            self.num_gc_graphs += batch.num_graphs

        elif task == "nc":
            num_train_nodes = batch.train_mask.sum().item()
            num_test_nodes = batch.batch.size(0) - num_train_nodes
            if train:
                self.num_nc_nodes += num_train_nodes
                return num_train_nodes
            else:
                self.num_nc_nodes += num_test_nodes
                return num_test_nodes


        elif task == "lp":
            self.num_lp_batches += 1
            num_lp_edges = batch.neg_edge_index.size(1) + batch.pos_edge_index.size(1)
            self.num_lp_edges += num_lp_edges
            return num_lp_edges

    def updatecosmos(self, task, batch, loss, acc, ray_index, num_test_nodes, num_lp_edges):
        if task == "gc":
            self.task_losses_all[task][ray_index].append((loss * batch.num_graphs).item())
            self.task_correct_all[task][ray_index].append((acc * batch.num_graphs).item())
        elif task == "nc":
            self.task_losses_all[task][ray_index].append((loss * num_test_nodes).item())
            self.task_correct_all[task][ray_index].append((acc * num_test_nodes).item())

        elif task == "lp":
            self.task_losses_all[task][ray_index].append((loss * num_lp_edges).item())
            self.task_correct_all[task][ray_index].append(acc.item())

    def get_average_stats(self):
        stats = {}
        for task in self.tasks:
            if len(self.task_correct[task]) == 0:
                continue

            stats[task] = {}
            if task == "gc":
                stats[task]['acc'] = np.stack(self.task_correct[task]).sum() / self.num_gc_graphs
                stats[task]['loss'] = np.stack(self.task_losses[task]).sum() / self.num_gc_graphs
            elif task == "nc":
                stats[task]['acc'] = np.stack(self.task_correct[task]).sum() / self.num_nc_nodes
                stats[task]['loss'] = np.stack(self.task_losses[task]).sum() / self.num_nc_nodes
            elif task == "lp":
                stats[task]['acc'] = np.stack(self.task_correct[task]).sum() / self.num_lp_batches
                stats[task]['loss'] = np.stack(self.task_losses[task]).sum() / self.num_lp_edges

        return stats

    def get_average_stats_cosmos(self, args):
        stats = {}

        for task in args.tasks:
            stats[task] = {}
            for ss in range(self.n_test_rays):
                if len(self.task_correct_all[task][ss]) == 0:
                    continue
                if task == "gc":
                    aa = np.stack(self.task_correct_all[task][ss]).sum() / self.num_gc_graphs
                    bb = np.stack(self.task_losses_all[task][ss]).sum() / self.num_gc_graphs
                    if ss == 0:
                        stats[task]['acc'] = aa
                        stats[task]['loss'] = bb
                    if aa > stats[task]['acc']:
                        stats[task]['acc'] = aa
                        stats[task]['loss'] = bb
                elif task == "nc":
                    aa = np.stack(self.task_correct_all[task][ss]).sum() / self.num_nc_nodes
                    bb = np.stack(self.task_losses_all[task][ss]).sum() / self.num_nc_nodes
                    if ss == 0:
                        stats[task]['acc'] = aa
                        stats[task]['loss'] = bb
                    if aa > stats[task]['acc']:
                        stats[task]['acc'] = aa
                        stats[task]['loss'] = bb
                elif task == "lp":
                    aa = np.stack(self.task_correct_all[task][ss]).sum() / self.num_lp_batches
                    bb = np.stack(self.task_losses_all[task][ss]).sum() / self.num_lp_edges
                    if ss == 0:
                        stats[task]['acc'] = aa
                        stats[task]['loss'] = bb
                    if aa > stats[task]['acc']:
                        stats[task]['acc'] = aa
                        stats[task]['loss'] = bb

        return stats

def print_task_accs_and_losses(tasks_epoch_stats):
    str_stats = ""
    for task in tasks_epoch_stats:
        task_acc = tasks_epoch_stats[task]['acc']
        task_loss = tasks_epoch_stats[task]['loss']
        str_stats += f"{task:>4}:{task_loss:^10.4f}|{task_acc:^12.4f}\n"
    tqdm.write(str_stats)


def log_task_accs_and_losses(tasks_epoch_stats):
    str_stats = ""
    for task in tasks_epoch_stats:
        task_acc = tasks_epoch_stats[task]['acc']
        task_loss = tasks_epoch_stats[task]['loss']
        str_stats += f"{task:>4}:{task_loss:^10.4f}|{task_acc:^12.4f}\n"
    return str_stats

def log_task_hv_and_point(test_hv):
    scores = test_hv["scores"]
    hv = test_hv["hv"]

    str_stats = f"hv:{hv:^10.4f}\n Scores:\n"
    for s in scores:
        for p, j in enumerate(s):
            if p != len(s)-1:
                str_stats += f"{j:^10.4f}\t"
            else:
                str_stats += f"{j:^10.4f}\n"
    return str_stats

def task_accs_and_losses(tasks_epoch_stats):
    task_accs_sum = 0.0
    task_accs = {}
    for task in tasks_epoch_stats:
        task_acc = tasks_epoch_stats[task]['acc']
        task_accs[task] = task_acc
        task_accs_sum = task_accs_sum + task_acc
    return task_accs_sum, task_accs

def dict_add(A, B):
    for key, value in B.items():
        if key in A:
            A[key] += value
        else:
            A[key] = value
    return dict(sorted(A.items(), key=lambda d:d[1]))

def dict_di(A, B):
    # A is dict, B is int,float~
    for key, value in A.items():
        A[key] = value/B
    return A

def print_train_epoch_stats(epoch, tasks_epoch_stats):
    tqdm.write("Epoch: {}\nTask:   Loss   |  Accuracy  \n".format(epoch))
    print_task_accs_and_losses(tasks_epoch_stats)


def print_test_stats(tasks_epoch_stats):
    tqdm.write("\n\n--- Results on Test Data:\nTask:   Loss   |  Accuracy \n")
    print_task_accs_and_losses(tasks_epoch_stats)
    tqdm.write("\n\n")

def print_val_stats(tasks_epoch_stats):
    tqdm.write("\n\n--- Results on Valid Data:\nTask:   Loss   |  Accuracy \n")
    print_task_accs_and_losses(tasks_epoch_stats)
    tqdm.write("\n\n")