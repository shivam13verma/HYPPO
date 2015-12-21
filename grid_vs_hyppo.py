'''
@author: adarsh, shivam
'''
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
from hypop import BayesOptCV
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
min_C = 6.4e-05
max_C = 60


def get_data(fpath):
    data = load_svmlight_file(fpath)
    return data[0], data[1]

train_path = 'splice_data/splice_noise_train.txt'
test_path = 'splice_data/splice_noise_test.txt'

train_data, train_labels = get_data(train_path)
test_data, test_label = get_data(test_path)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))

train_data = scaler.fit_transform(train_data.toarray())
test_data = scaler.transform(test_data.toarray())

print train_data.shape
print test_data.shape

def svccv(C):
    return cross_val_score(SVC(C=C, kernel='poly', degree=3, random_state=2),
                           train_data, train_labels, 'accuracy', cv=5).mean()

def do_grid_search_for(n):
    params = {"C":np.linspace(min_C, max_C, n)}
    gs = GridSearchCV(SVC(kernel='poly', degree=3, random_state=2), param_grid=params, cv=5, n_jobs=-1, scoring='accuracy')
    gs.fit(train_data, train_labels)
    return gs.best_score_

def do_random_search_for(n):
    params = {"C":np.linspace(min_C, max_C, 100)}
    gs = RandomizedSearchCV(SVC(kernel='poly', degree=3, random_state=2), param_distributions=params, cv=5, n_jobs=-1, scoring='accuracy', n_iter=n)
    gs.fit(train_data, train_labels)
    return gs.best_score_


def do_bayes_opt_for(n):
    svcBO = BayesOptCV(svccv, param_grid={'C':{'type':'float', 'min':min_C, 'max':max_C}},
                   bigger_is_better=True, verbose=2)

#     svcBO.initialize(num_init=10, init_grid={'C': [1, 15, 60]})
    svcBO.initialize(num_init=3)
    kernel_param = {'theta0':0.5}
    acqui_param = {'kappa':2}
    svcBO.optimize(kernel_param=kernel_param, acqui_param=acqui_param, n_iter=n - 3, acqui_type='pi', n_acqui_iter=200)
    return svcBO.get_best()

grid_errs = []
rs_errs = []
bayes_opt_errs = []
Ns = range(5, 6)
n_i = 2
for i in range(n_i):
    print '***********************************'
    print str(i)
    
    for n in Ns:
        grid_errs.append(1 - do_grid_search_for(n))
        print 'Grid Done'
        rs_errs.append(1 - do_random_search_for(n))
        print 'Random done'
        bayes_opt_errs.append(1 - do_bayes_opt_for(n))
        print 'Bayes done'

grid_errs = np.array(grid_errs).reshape(n_i, -1)
rs_errs = np.array(rs_errs).reshape(n_i, -1)
bayes_opt_errs = np.array(bayes_opt_errs).reshape(n_i, -1)

grid_errs_means = grid_errs.mean(axis=0)
rs_errs_means = rs_errs.mean(axis=0)
bayes_opt_err_means = bayes_opt_errs.mean(axis=0)

grid_errs_stds = grid_errs.std(axis=0)
rs_errs_stds = rs_errs.std(axis=0)
bayes_opt_err_stds = bayes_opt_errs.std(axis=0)
    
pickle.dump(grid_errs, open('grid_errs.dump', 'wb'))
pickle.dump(rs_errs, open('rs_errs.dump', 'wb'))
pickle.dump(bayes_opt_errs, open('bayes.dump', 'wb'))

pickle.dump(grid_errs_means, open('grid_errs_means.dump', 'wb'))
pickle.dump(rs_errs_means, open('rs_errs_means.dump', 'wb'))
pickle.dump(bayes_opt_err_means, open('bayes_means.dump', 'wb'))

pickle.dump(grid_errs_stds, open('grid_errs_stds.dump', 'wb'))
pickle.dump(rs_errs_stds, open('rs_errs_stds.dump', 'wb'))
pickle.dump(bayes_opt_err_stds, open('bayes_stds.dump', 'wb'))


plt.errorbar(Ns, grid_errs_means, yerr=grid_errs_stds, fmt='o-', color='r')
plt.errorbar(Ns, rs_errs_means, yerr=rs_errs_stds, fmt='o-', color='g')
plt.errorbar(Ns, bayes_opt_err_means, yerr=bayes_opt_err_stds, fmt='o-', color='b')
plt.legend(['Grid Search', 'Random Search', 'HYPPO'])

plt.savefig('Hyppo.pdf')
plt.savefig('Hyppo.png')
