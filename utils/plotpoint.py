import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class PointScatter:

    def __init__(self, Dimension, grid=True, legend=True, title=None, coordLabels=None, saveName=None):
        if title is None:
            title = 'ParCoordPlot'
        if coordLabels is None:
            coordLabels = ['F1', 'F2', 'F3']
        # check Dimension
        if Dimension != 2 and Dimension != 3:
            raise RuntimeError('Error in PointScatter.py: M must be 2 or 3. ')
        self.Dimension = Dimension
        self.grid = grid
        self.legend = legend
        self.title = title
        self.coordLabels = coordLabels
        self.saveName = saveName
        self.data_set = []  
        self.params_set = []  
        self.plots_set = []
        self.history = [] 
        self.fig = None
        self.ax = None

    def add(self, points, marker='o', objectSize=5, color='blue', alpha=1.0, label=None):
        """
        """
        if points is None:
            return
        if not isinstance(points, np.ndarray):
            raise RuntimeError(
                'Error in PointScatter.py: The type of the points must be numpy ndarray. ')
        if points.ndim == 1:
            points = np.array([points])
            if points.shape[1] != self.Dimension:
                raise RuntimeError(
                    'Error in PointScatter.py: The length of the points must be equal to Dimension if its dimension is 1. ')
        elif points.ndim == 2:
            if points.shape[0] == 0:
                return
            if points.shape[1] != self.Dimension:
                raise RuntimeError(
                    'Error in PointScatter.py: The number of the column of the points must be equal to Dimension if its dimension is 2. ')
        self.data_set.append(points)
        self.params_set.append(
            {'marker': marker, 'objectsize': objectSize, 'color': color, 'alpha': alpha, 'label': label})
        self.history += [{'data_set': self.data_set, 'params_set': self.params_set}]

    def draw(self):

        if self.Dimension == 2:
            if self.fig is None and self.ax is None:
                self.fig, self.ax = plt.subplots() 
            for idx, data in enumerate(self.data_set):
                params = self.params_set[idx]
                plot = self.ax.plot(data[:, 0], data[:, 1], params['marker'], markersize=params['objectsize'],
                                    linewidth=params['objectsize'], color=params['color'], alpha=params['alpha'],
                                    label=params['label'])
                self.plots_set.append(plot)
                self.ax.set_xlabel(self.coordLabels[0])
                self.ax.set_ylabel(self.coordLabels[1])
        elif self.Dimension == 3:
            if self.fig is None and self.ax is None:
                self.fig = plt.figure()
                self.ax = Axes3D(self.fig) 
                self.ax.view_init(elev=30, azim=45) 
            for idx, data in enumerate(self.data_set):
                params = self.params_set[idx]
                plot = self.ax.plot(data[:, 0], data[:, 1], data[:, 2], params['marker'],
                                    markersize=params['objectsize'], color=params['color'], alpha=params['alpha'],
                                    label=params['label'])
                self.plots_set.append(plot)
                self.ax.set_xlabel(self.coordLabels[0])
                self.ax.set_ylabel(self.coordLabels[1])
                self.ax.set_zlabel(self.coordLabels[2])
        if self.title is not None:
            self.ax.set_title(self.title)
        if self.legend:
            plt.legend()
        plt.grid(self.grid)
        plt.draw()

    def refresh(self):
        if self.fig and self.ax:
            plt.pause(1 / 24)
            self.ax.cla()
        self.data_set = []
        self.params_set = []
        self.plots_set = []

    def show(self):
        if self.saveName is not None:
            self.fig.savefig(self.saveName + '.svg', dpi=300, bbox_inches='tight')
        # plt.show()

    def createAnimation(self, fps=6):


        def update(i, plotObject):
            plotObject.ax.cla()
            plotObject.data_set = plotObject.history[i]['data_set']
            plotObject.params_set = plotObject.history[i]['params_set']
            plotObject.draw()

        if len(self.history) > 0:
            if self.fig is None and self.ax is None:
                if self.Dimension == 2:
                    self.fig, self.ax = plt.subplots()  
                elif self.Dimension == 3:
                    self.fig = plt.figure()  
                    self.ax = Axes3D(self.fig) 
                    self.ax.view_init(elev=30, azim=45)  
            print('Creating gif...')
            anim = animation.FuncAnimation(self.fig, update, frames=len(self.history), fargs=(self,))
            anim.save(self.title + '.gif', fps=fps)
            print('gif has been created.')

    def close(self):
        plt.close()
        for item in self.history:
            item.clear()
        self.history = []
        self.data_set = []
        self.params_set = []
        self.plots_set = []