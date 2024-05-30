import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch

def circle_points(n, min_angle=0.1, max_angle=np.pi / 2 - 0.1, dim=2):
    # generate evenly distributed preference vector
    assert dim > 1
    if dim == 2:
        ang0 = 1e-6 if min_angle is None else min_angle
        ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
        angles = np.linspace(ang0, ang1, n, endpoint=True)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.c_[x, y]
    elif dim == 3:
        # Fibonacci sphere algorithm
        # https://stackoverflow.com/a/26127012
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

        n = n*8   # we are only looking at the positive part
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            if x >=0 and y>=0 and z>=0:
                points.append((x, y, z))
        return np.array(points)
    else:
        # this is an unsolved problem for more than 3 dimensions
        # we just generate random points
        points = np.random.rand(n, dim)
        points /= points.sum(axis=1).reshape(n, 1)
        return points

def getNumParams(params):
    numParams, numTrainable = 0, 0
    for param in params:
        npParamCount = np.prod(param.data.shape)
        numParams += npParamCount
        if param.requires_grad:
            numTrainable += npParamCount
    return numParams, numTrainable

class ParetoFront():

    def __init__(self, labels, logdir='tmp', prefix=""):
        self.labels = labels
        self.logdir = os.path.join(logdir, 'pf')
        self.prefix = prefix
        self.points = np.array([])

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def append(self, point):
        point = np.array(point)
        if not len(self.points):
            self.points = point
        else:
            self.points = np.vstack((self.points, point))

    def plot(self):
        p = self.points
        plt.plot(p[:, 0], p[:, 1], 'o')
        plt.xlabel(self.labels[0])
        plt.ylabel(self.labels[1])
        plt.savefig(os.path.join(self.logdir, "x_{}.png".format(self.prefix)))
        plt.close()

def get_runname(settings):
    slurm_job_id = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ and 'hpo' not in settings['logdir'] else None
    if slurm_job_id:
        runname = f"{slurm_job_id}"
        if 'ablation' in settings['logdir']:
            runname += f"_{settings['lamda']}_{settings['alpha']}"
    else:
        runname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if 'task_id' in settings:
        runname += f"_{settings['task_id']:03d}"
    return runname

def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def save_point(point, name):
    """ Saves the point. """
    filename = os.path.join('{}.txt'.format(name))
    with open(filename, 'w') as f:
        for i in point:
            for j in i:
                f.write(f"{j:>7.4f}\t")
            f.write("\n")

def save(model, model_path):
  torch.save(model.state_dict(), model_path)