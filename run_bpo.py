from __future__ import print_function
from __future__ import division

from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

from hypop import BayesOptCV

# Load data set and target values
data, target = make_classification(n_samples=2500,
                                   n_features=45,
                                   n_informative=12,
                                   n_redundant=7)

def svccv(C, gamma):
    return cross_val_score(SVC(C=C, gamma=gamma, random_state=2),
                           data, target, 'f1', cv=5).mean()

def rfccv(n_estimators, min_samples_split, max_features):
    return cross_val_score(RFC(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),
                               random_state=2),
                           data, target, 'f1', cv=5).mean()

if __name__ == "__main__":

    svcBO = BayesOptCV(svccv, param_grid={'C':{'type':'float', 'min':0.02, 'max':20}, 'gamma':{'type':'float', 'min':0.05, 'max':5}}, verbose=2)
    svcBO.initialize(num_init=3, init_grid={'C': [0.001, 0.01, 0.1], 'gamma': [0.001, 0.01, 0.1]})

    rfcBO = BayesOptCV(rfccv, param_grid={'n_estimators': {'type':'int', 'min':10, 'max':250},
                                         'min_samples_split': {'type':'int', 'min':2, 'max':25},
                                         'max_features': {'type':'int', 'min':0.1, 'max':0.999}}, verbose=2)
    kernel_param = {}  # something
    acqui_param = {}  # something
    rfcBO.initialize(num_init=3)
    svcBO.optimize(kernel_param=kernel_param, acqui_param=acqui_param)
    rfcBO.optimize(kernel_param=kernel_param, acqui_param=acqui_param)

    print('Final Results')
    print('SVC: %f' % svcBO.report['best']['best_val'])
    print('RFC: %f' % rfcBO.report['best']['best_val'])
