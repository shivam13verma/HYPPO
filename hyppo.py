# -*- coding: utf-8 -*-
"""
filename: hyppo.py

@author: shivamverma, adarshjois
"""
from __future__ import division

# from gaussian_process import GaussianProcess as GP 
# from sklearn.gaussian_process import GaussianProcess as GP
from gaussian_process_sklearn.gaussian_process import GaussianProcess as GP
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as sns

class BayesOptCV(object):
    """
        Performs Gaussian Process-based Bayesian Optimization for optimizing model hyperparameters.
    """

    def __init__(self, fun, param_grid, verbose=1, if_noise=True, bigger_is_better=True, obj='', if_plot=True):
        
        self.param_grid = param_grid
        self.param_num = len(param_grid)
        self.fun = fun
        self.verbose = verbose
        self.if_noise = if_noise
        self.param_keys = param_grid.keys()
        self.param_lims = []
        for key in self.param_grid.keys():
            self.param_lims.append((self.param_grid[key]['min'], self.param_grid[key]['max']))
        self.param_lims = np.array(self.param_lims)
        self.if_init = False
        self.init_x = []
        self.init_y = []
        # self.num_init=0
        self.num_eval = 0  # stores number of function evaluations
        self.X = []
        self.Y = []
        self.report = {}
        self.bigger_is_better = bigger_is_better
        self.obj_xtest = np.linspace(self.param_lims[0][0], self.param_lims[0][1], 50) #for plotting
        self.if_plot = if_plot
        # self.obj = [obj(x_test) for x_test in self.obj_xtest]


    def initialize(self, num_init, init_grid={}):
        """
        Initializes self.init_grid for GP
        """
        if bool(init_grid):
            temp = []
            temp_length = []
            for key in self.param_keys:
                temp.append(init_grid[key])
                temp_length.append(len(init_grid[key]))
            if all([el == temp_length[0] for el in temp_length]):
                pass
            else:
                raise ValueError('Number of initializations for all parameters should be the same.')
        else:
            temp = [np.random.uniform(x[0], x[1], size=num_init) for x in self.param_lims]
        
        self.init_x = list(map(list, zip(*temp)))
#         init_x = list(map(list, zip(*init_grid.values())))
        self.init_y = [];
        for x in self.init_x:
            self.init_y.append(self.fun(**dict(zip(self.param_keys, x))))
            self.num_eval += 1
        self.X = np.array(self.init_x)
        self.Y = np.array(self.init_y)
        print self.X
        print self.Y
        self.if_init = True
        
    def remove_gp_duplicates(self, x):
        
        order = np.lexsort(x.T)
        new_order = np.argsort(order)
    
        x = x[order]
        diff = np.diff(x, axis=0)
        dedup = np.ones(len(x), 'bool')
        dedup[1:] = (diff != 0).any(axis=1)
        return dedup[new_order]

    def maximize_acquisition(self, acqui_fun, gp, y_max, n_acqui_iter=25):
        """
        Maximizes acquisition function for given gp.
        
        Parameters:
        acqui_fun: Function of type acquisition_function, initialized with acqui_type, kappa, eta etc.
        gp: pre-initialized Gaussian Process
        y_max: Current best (max) value of function
        n_iter: Number of times to run maximization (for best result)
        
        Returns:
        x_best: New best (max) value of x, at which fun attains best value
        """
        # initialize
        x_best = self.param_lims[:, 0] 
        ei_best = 0
        n_iter = 0
        is_duplicate = True
        # iterate
        for i in range(n_acqui_iter):
            x_init = np.array([np.random.uniform(x[0], x[1], size=1) for x in self.param_lims]).T
            
            min_xy = minimize(lambda x:-acqui_fun(gp=gp, x=x.reshape(1, -1), y_max=y_max), x_init, bounds=self.param_lims, method='L-BFGS-B')            
            if -min_xy.fun >= ei_best:
                x_best = min_xy.x
                ei_best = -min_xy.fun
#                 else:
#                     min_xy = minimize(lambda x:acqui_fun(gp=gp, x=x.reshape(1, -1), y_max=y_max), x_init, bounds=self.param_lims, method='L-BFGS-B')
#                     if min_xy.fun <= ei_best:
#                         x_best = min_xy.x
#                         ei_best = min_xy.fun
                        
        return x_best, ei_best
#    
#        return dedup[new_order]


# # optimize acquisition function
    def optimize(self, acqui_param, kernel_param, n_iter=25, acqui_type='pi', n_acqui_iter=50, kernel_type='squared_exponential'):
        """
        Core bayesian optimization using gaussian processes.
        
        Parameters:
        
        
        Returns:
        """ 
        import time
        timestr = time.strftime("%Y%m%d%H%M")
        if not self.if_init:
            self.initialize(num_init=5, init_grid={})
        # for custom GP
        gp = GP(corr=kernel_type, **kernel_param)
        ur = self.remove_gp_duplicates(self.X)
        gp.fit(self.X[ur], self.Y[ur])
        
        # for sklearn GP
#        gp = GP(theta0=np.random.uniform(0.001, 0.05, len(self.param_keys)),
#                random_start=25)
#        gp.set_params(**kernel_param)
#        gp.fit(self.X, self.Y)  
        
        acqui = acquisition_function(acqui_type=acqui_type, **acqui_param)
        acqui_fun = acqui.set_function()
        y_max = self.Y.max()
        
        # For duplicate case
        # new_ind = remove_gp_duplicates(self.X)
        # gp.fit(self.X[new_ind], self.Y[new_ind])
        
        x_max, ei_best = self.maximize_acquisition(acqui_fun=acqui_fun, gp=gp, y_max=y_max, n_acqui_iter=n_acqui_iter)


        for i in range(n_iter):
            self.X = np.concatenate((self.X, x_max.reshape((1, len(self.param_keys)))), axis=0)
            self.Y = np.append(self.Y, self.fun(**dict(zip(self.param_keys, x_max))))
            ur = self.remove_gp_duplicates(self.X)
            print self.X
            gp.fit(self.X[ur], self.Y[ur])
            if self.if_plot:
                sns.figure(figsize=(20,10))
                mus_and_covs = [gp.predict(x_test, eval_MSE=True) for x_test in self.obj_xtest]
                self.plot_gp(mus_and_covs)
             # diag_cov = np.diag(covs).reshape(-1, 1)
                 
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
            x_max, ei_best = self.maximize_acquisition(acqui_fun=acqui_fun, gp=gp, y_max=y_max, n_acqui_iter=n_acqui_iter)
            if self.verbose >= 2:
                print "Best x at round " + str(i) + " is: " + str(x_max)
                print "Best y at round " + str(i) + " is: " + str(y_max)
                if self.bigger_is_better:
                    print 'best so far {}'.format(self.Y.max())
                    print 'params{}'.format(dict(zip(self.param_keys, self.X[self.Y.argmax()])))
                else:
                    print 'best so far {}'.format(self.Y.min())
                    print 'params{}'.format(dict(zip(self.param_keys, self.X[self.Y.argmin()])))    
            if self.if_plot:
                self.plot_acquisition(acqui_fun, x_max, gp, y_max, ei_best, oned_index=0)
                #sns.show()
                sns.title('t = '+str(i),fontsize=18)
                #my_dpi=96
                #sns.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
                #sns.savefig('my_fig.pdf', dpi=my_dpi)
                sns.savefig('/Users/shivamverma/Documents/HYPPO/plots/Hyppo_comparison_'+timestr+'_'+str(i)+'_'+'_.pdf')
                sns.savefig('/Users/shivamverma/Documents/HYPPO/plots/Hyppo_comparison_'+timestr+'_'+str(i)+'_'+'_.png')
        if self.bigger_is_better:
            self.report['best'] = {'best_val': self.Y.max(), 'best_param': dict(zip(self.param_keys, self.X[self.Y.argmax()]))}
        else:
            self.report['best'] = {'best_val': self.Y.min(), 'best_param': dict(zip(self.param_keys, self.X[self.Y.argmin()]))}
        
        self.report['all'] = {'values': [], 'params': []}
        for b, a in zip(self.Y, self.X):
            self.report['all']['values'].append(b)
            self.report['all']['params'].append(dict(zip(self.param_keys, a)))

        
        if self.verbose >= 1:
            print "Best params are: " + str(self.report['best'])
    
    def get_best(self):
        return self.Y.max()
        
    def plot_gp(self, mus_and_covs):
        mus = [1 - tup[0] for tup in mus_and_covs]
        diag_cov = [tup[1] for tup in mus_and_covs]
        Xtest = self.obj_xtest
        sns.subplot(211)
        sns.plot(self.X, 1.0 - self.Y, 'r+', ms=15)
        sns.plot(Xtest.flat, mus,'k',linewidth=2)
        #sns.savefig('/Users/shivamverma/Desktop/cvplot.pdf')
        # sns.plot(Xtest, self.obj, 'b-')
    
        sns.gca().fill_between(Xtest.flat, (mus - 3 * np.sqrt(diag_cov)).flat,
                          (mus + 3 * np.sqrt(diag_cov)).flat,
                          where=None, color="#dddddd")
        #sns.xlabel('C')
        sns.ylabel('CV Error',fontsize=18)
        #plt.legend((line1, line2, line3), ('label1', 'label2', 'label3'))
        sns.legend(['Observations','Mean','Variance'],loc='center left', bbox_to_anchor=(1, 0.5),fontsize=12)
    
    def plot_acquisition(self, acqui_fun, x_max, gp, y_max, ei_best, oned_index=0, twod_index=0, if2D=False):
        if not if2D:
            sns.subplot(212)
#            ei_acqui = acquisition_function(acqui_type='ei')
#            ei_acqui_fun = ei_acqui.set_function()
#            pi_acqui = acquisition_function(acqui_type='pi')
#            pi_acqui_fun = pi_acqui.set_function()
#            ei_x_max, ei_ei_best = self.maximize_acquisition(acqui_fun=ei_acqui_fun, gp=gp, y_max=y_max, n_acqui_iter=200)
#            pi_x_max, pi_ei_best = self.maximize_acquisition(acqui_fun=pi_acqui_fun, gp=gp, y_max=y_max, n_acqui_iter=200)
##            
            x_p = np.linspace(self.param_lims[oned_index][0], self.param_lims[oned_index][1], 100)
            y_p = np.array([acqui_fun(gp=gp, x=x0.reshape(1, -1), y_max=y_max) for x0 in x_p])
            
#            ei_x_p = np.linspace(self.param_lims[oned_index][0], self.param_lims[oned_index][1], 100)
#            ei_y_p = np.array([ei_acqui_fun(gp=gp, x=x0.reshape(1, -1), y_max=y_max) for x0 in x_p])
#            pi_x_p = np.linspace(self.param_lims[oned_index][0], self.param_lims[oned_index][1], 100)
#            pi_y_p = np.array([pi_acqui_fun(gp=gp, x=x0.reshape(1, -1), y_max=y_max) for x0 in x_p])
##            
            sns.plot(x_p, y_p.flatten(),linewidth=2) #UCB
#            sns.plot(ei_x_p, ei_y_p.flatten())
#            sns.plot(pi_x_p, pi_y_p.flatten())

            
            ei_best2 = acqui_fun(gp=gp, x=x_max.reshape(1, -1), y_max=y_max)
            sns.plot(x_max, ei_best, 'ro',ms=15)
#            sns.plot(ei_x_max, ei_ei_best, 'ro')
#            sns.plot(pi_x_max, pi_ei_best, 'ro')
#            sns.title('Cross Validation Error vs. C')
            sns.xlabel('C',fontsize=18)
            sns.ylabel('Acquisition function',fontsize=18)
            #plt.legend((line1, line2, line3), ('label1', 'label2', 'label3'))
            sns.legend(['GP-UCB'],loc='center left', bbox_to_anchor=(1, 0.5),fontsize=12)
#            sns.legend(['GP-UCB','EI','PI'],loc='center left', bbox_to_anchor=(1, 0.5))

#    def get_best_model(self, X, y):
#        train_X, train_y = X, y
#        metrics = self.test_scores
#        list_of_dict = self.param_dict_list
#        if self.verbose in [1,2]:
#            print [i/float(self.cv) for i in metrics]
#        if self.bigger_is_better:
#            bestIndex = np.argmax(metrics)
#        else:
#            bestIndex = np.argmin(metrics)
#        bestParam = list_of_dict[bestIndex]
#        bm = self.estimator(**bestParam)
#        bestModel = bm.fit(train_X, train_y)
#        if self.verbose in [1,2]:
#            print "Best parameters are: ", bestParam
#        return bestModel

#    def get_best_param(self):
#        emp={}
#        for i in self.report[0].keys():
#            emp[i]=0
#        for i in self.report.values():
#            #print i
#            for k in range(len(emp)):
#                emp[i.keys()[k]] += [j[1][0] for j in i.values()][k]
#        print emp
#        if self.verbose in [1,2]:
#            print emp.keys()
#            print [i/float(self.cv) for i in emp.values()]
#        if self.bigger_is_better:
#            best_param = max(emp)
#        else:
#            best_param = min(emp)
#    
#        #bestParam = list_of_dict[bestIndex]
#        #bm = self.estimator(**bestParam)
#        #bestModel = bm.fit(train_X, train_y)
#        if self.verbose in [1,2]:
#            print "Best parameters are: ", best_param
#        return best_param


class acquisition_function(object):
    """
    Acquisition function for performing Bayesian Optimization. Includes PI, EI, GP-UCB.
    """
    
    def __init__(self, acqui_type='ei', eta=0.01, kappa=2):
        self.acqui_type = acqui_type
        self.eta = eta
        self.kappa = kappa
        
    def set_function(self):
        if self.acqui_type in ['ei', 'expected improvement']:
            return self.EI
        if self.acqui_type in ['pi', 'poi', 'probability of improvement']:
            return self.PI
        if self.acqui_type in ['gp-ucb', 'ucb']:
            return self.GP_UCB
        raise ValueError('Acquisition function should be EI, PI or GP-UCB.')
        

    def EI(self, gp, x, y_max):
        mu, var = gp.predict(x, eval_MSE=True)
        if var == 0:
            return 0
        else:
            Z = (mu - y_max - self.eta) / np.sqrt(var)
            ei = (mu - y_max - self.eta) * norm.cdf(Z) + np.sqrt(var) * norm.pdf(Z)
            return ei

    def GP_UCB(self, gp, x, y_max):
        mu, var = gp.predict(x, eval_MSE=True)
        return mu + self.kappa * np.sqrt(var)

    def PI(self, gp, x, y_max):
        mu, var = gp.predict(x, eval_MSE=True)
        if var == 0:
            return 1
        else:
            Z = (mu - y_max - self.eta) / np.sqrt(var)
            return norm.cdf(Z)




