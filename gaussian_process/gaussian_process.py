from __future__ import division

import numpy as np
import numpy.ma as ma
import scipy.spatial.distance as dist

f = lambda x: 3 * x + 9


class GaussianProcess(object):
    '''
    Class used to create a Gaussian process
    to fit things.
    '''
    kernel_types = set(['exp', 'sq_exp', 'squared_exponential' , 'spherical', 'linear'])

    def __init__(self, corr='sq_exp', kernel_params={'tau': 1.0, 'l': 1.0}, sigma=0.00005):
        self.kernel = corr
        self.kernel_params = kernel_params
        self.sigma = sigma

    def computeKernel(self, X1, X2):
        if self.kernel == 'sq_exp'or self.kernel == 'squared_exponential':
            if not 'tau'in self.kernel_params or not 'l' in self.kernel_params:
                raise ValueError(
                    "Please check kernel params for 'tau' and 'l'")
            d = dist.cdist(X1, X2, 'sqeuclidean')
            tau = self.kernel_params['tau']
            l = self.kernel_params['l']
            k = (tau ** 2) * np.exp(-0.5 / (l ** 2) * d)
        elif self.kernel == 'exp':
            if not 'tau'in self.kernel_params or not 'l' in self.kernel_params:
                raise ValueError(
                    "Please check kernel params for 'tau' and 'l'")
            d = dist.cdist(X1, X2, 'euclidean')
            tau = self.kernel_params['tau']
            l = self.kernel_params['l']
            k = (tau ** 2) * np.exp(-0.5 / l * d)
        elif self.kernel == 'spherical':
            tau = -1
            theta = -1
            if not 'tau' in self.kernel_params:
                tau = 1.0
            else:
                tau = self.kernel_params['tau']

            if not 'theta' in self.kernel_params:
                theta = 1.0
            else:
                theta = self.kernel_params['theta']

            d = dist.cdist(X1, X2, 'euclidean')
            mask = d <= theta
            d = ma.MaskedArray(d, ~mask)
            d3 = d ** 3
            k = (tau ** 2) * (1 - (1.5 / theta) * d + (0.5) * d3 / (theta ** 3))
            k = ma.getdata(k.filled(0.0))
        elif self.kernel == 'linear':
            sigma = -1
            tau = -1
            c = np.zeros(X1.shape[1])
            if not 'sigma' in self.kernel_params:
                sigma = 1.0
            else:
                sigma = self.kernel_params['sigma']
            if not 'tau' in self.kernel_params:
                tau = 1.0
            else:
                tau = self.kernel_params['tau']
            if not 'c' in self.kernel_params:
                pass
            else:
                c = self.kernel_params['c']
            d = (X1 - c).dot((X2 - c).T)
            k = (sigma ** 2) + (tau ** 2) * d
        elif self.kernel == 'matern32':
            d = dist.cdist(X1, X2, 'euclidean')
            K = d * np.sqrt(3)
            k = (1. + K) * np.exp(-K)
        elif self.kernel == 'matern52':
            d = dist.cdist(X1, X2, 'euclidean')
            K = d * np.sqrt(5)
            k = (1. + K + K ** 2 / 3.0) * np.exp(-K) 

        return k

    def update(self, train_data, train_vals):
        self.X = np.vstack([self.xtrain, train_data])
        self.y = np.vstack([self.ytrain, train_vals])
        self.fit(self.X, self.y)

    def fit(self, X, y):
        self.X = X
        self.trainSize = self.X.shape[0]
        self.y = y
        # K(X,X)
        self.K = self.computeKernel(
            self.X, self.X) + (self.sigma ** 2) * np.eye(self.trainSize)
        self.__alpha = np.linalg.lstsq(
            self.K, self.y)[0]

    def __sqrt(self, M):
        u, s, _ = np.linalg.svd(M)
        # s = np.around(s, decimals=6)
        return np.dot(u, np.diag(np.sqrt(s)))

    def sample(self, x_test, ns):
        self.predict(x_test)
        f_sampled = self.mus + \
            np.dot(
                self.__sqrt_covs, np.random.normal(size=(x_test.shape[0], ns)))
        return f_sampled

    def predict(self, x_test, eval_MSE=True):
        # Compute the mean
        # equivalent to K(X*,X)
        K_ = self.computeKernel(x_test, self.X)
        self.mus = K_.dot(self.__alpha)
        # Compute the covariances
        if eval_MSE == True:
            K__ = self.computeKernel(x_test, x_test)
            W = np.linalg.lstsq(self.K, K_.T)[0]
            self.covs = K__ - K_.dot(W)

        # use this for sampling from the gaussian later.
            self.__sqrt_covs = self.__sqrt(self.covs)
            return self.mus, self.covs
        else:
            return self.mus


def demo_gp():
    N = 50  # number of training points.
    n = 20  # number of test points.
    sigma = 0.005  # noise variance.
    ns = 100  # number of samples

    X = np.random.uniform(-5, 5, size=(N, 1))
    y = f(X) + sigma * np.random.randn(N, 1)
    Xtest = np.linspace(-5, 5, n).reshape(-1, 1)
    gp = GaussianProcess(corr='matern52')
    gp.fit(X, y)
    print gp.predict(Xtest)
    print gp.sample(Xtest, 20)
if __name__ == '__main__':
    demo_gp()
