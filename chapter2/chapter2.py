#!/usr/bin/env python
'''
{
    'author'  : 'Robert M. Johnson',
    'website' : 'mattshomepage.com',
    'twitter' : '@mattshomepage',
    'github'  : 'mattsgithub'
}
'''

from matplotlib import pyplot
import numpy as np

class Point():
    def __init__(self, color, x1, x2):
        self.color = color
        self.x1 = x1
        self.x2 = x2


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
   
    f = lambda x: (0.5 - beta.item(0) - beta.item(1)*x ) / beta.item(2)
    X1 = [x1 for x1 in np.arange(X[:,1].min(), X[:,1].max(), 0.1)]
    X2 = [f(x1) for x1 in X1]

    for p in points:
        pyplot.plot(p.x1, p.x2, 'o', mfc = 'none', mec = p.color)

    pyplot.plot(X1, X2, color = 'black')

    pyplot.show()


def get_generated_points():
    points   = []
    points = []

    n      = 10 # number of samples
    cov    = np.eye(2) # generates "2x2" identity matrix
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


if __name__ == '__main__':
    plot_figure_2_1()


