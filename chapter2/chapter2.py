#!/usr/bin/env python
'''
{
    'author'  : 'Robert M. Johnson',
    'website' : 'mattshomepage.com',
    'twitter' : 'mattshomepage',
    'github'  : 'mattsgithub'
}
'''

from matplotlib import pyplot
import numpy as np
from sklearn import neighbors
from matplotlib.colors import ListedColormap

class Point():
    def __init__(self, color, x1, x2):
        self.color = color
        self.x1 = x1
        self.x2 = x2

def get_generated_points():
    '''
        On pages 16-17 it was revealed the author were generating data from a combination
        of multivariate normal distributions. This function returns these points.
    '''
    points   = []
    points = []
    
    n      = 10 # Number of samples
    cov    = np.eye(2) # Generates a "2x2" identity matrix
    means1 = np.random.multivariate_normal([1,0], cov, n)
    means2 = np.random.multivariate_normal([0,1], cov, n)
    
    N = range(1,100)
    cov = cov / 5
    
    for i in N:
        mu = means1[np.random.randint(n)]
        p = np.random.multivariate_normal(mu, cov, 1)[0]
        points.append( Point('blue', p[0], p[1]) )
        
        mu = means2[np.random.randint(n)]
        p = np.random.multivariate_normal(mu, cov, 1)[0]
        points.append( Point('orange', p[0], p[1]) )
    
    
    return points


def plot_figure_2_1():
    points = get_generated_points()
    
    X = []
    Y = []
    for p in points:
        X.append([1.0,p.x1,p.x2])
        Y.append([1 if p.color == 'blue' else 0])
    
    X = np.matrix(X)
    Y = np.matrix(Y)
    
    X_T = X.transpose()
    beta = np.linalg.inv(X_T * X) * X_T * Y
    
    decision_boundary = 0.5
    
    # This function solves for x2 given x1, y = 0.5, and beta
    f = lambda x: (decision_boundary - beta.item(0) - beta.item(1)*x ) / beta.item(2)
    
    X1 = [x1 for x1 in np.arange(X[:,1].min(), X[:,1].max(), 0.1)]
    X2 = [f(x1) for x1 in X1]
    
    for p in points:
        pyplot.plot(p.x1, p.x2, 'o', mfc = 'none', mec = p.color)
    
    pyplot.plot(X1, X2, color = 'black')
    
    pyplot.show()



def plot_figure_2_2(K=15):
    '''
        Nearest neighbor plot with K = 15
        Code for this function is adapted from:
        
        http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#example-neighbors-plot-classification-py
    '''
    
    points = get_generated_points()
    
    X = []
    Y = []
    for p in points:
        X.append([p.x1,p.x2])
        Y.append([1 if p.color == 'blue' else 0])
    
    # Transform into numpy matrices. Required by later computations.
    X = np.matrix(X)
    Y = np.matrix(Y)

    # Create color maps
    colors = ListedColormap(['#FFFAF5', '#F5F8FF'])

    # We create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(K)
    clf.fit(X, Y)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    

    # Ravel returns a flattened array and np.c_ concetenates both flattened arrays
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                         
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pyplot.figure()
    pyplot.pcolormesh(xx, yy, Z, cmap=colors)
     
    # Plot also the training points
    pyplot.xlim(xx.min(), xx.max())
    pyplot.ylim(yy.min(), yy.max())
    
    for p in points:
        pyplot.plot(p.x1, p.x2, 'o', mfc = 'none', mec = p.color)

    pyplot.show()


def plot_figure_2_3():
    '''
        This is a plot of Figure 2.2 with the exception of K = 1 instead of K = 15
    '''
    plot_figure_2_2(1)


def plot_figure_2_4():
    pass


def plot_figure_2_5():
    pass


def plot_figure_2_6():
    pass

def plot_figure_2_7():
    pass

def plot_figure_2_8():
    pass

def plot_figure_2_9():
    pass

def plot_figure_2_10():
    pass

def plot_figure_2_11():
    pass



if __name__ == '__main__':
    plot_figure_2_3()


