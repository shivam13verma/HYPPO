'''
@author: adarsh, shivam
'''

from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
from hypop import BayesOptCV
import matplotlib.pyplot as plt

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
import numpy as np
f = lambda x: np.sin(x)
sigma = 0.02  # noise variance.

def sin(C):
    
    return f(C) + sigma * np.random.randn(1)[0]
def svccv(C, gamma):
    return 1 - cross_val_score(SVC(C=C, kernel='rbf', gamma=gamma , random_state=2),
                           train_data, train_labels, 'f1', cv=5).mean()
                           
svcBO = BayesOptCV(svccv, param_grid={'C':{'type':'float', 'min':6.4e-05, 'max':60},
                                      'gamma':{'type':'float', 'min':6.4e-5, 'max':10}},
                   bigger_is_better=False, verbose=2)

svcBO.initialize(num_init=10, init_grid={'C': [1, 15, 60], 'gamma':[1, 5, 10]})
# kernel_param = {'nugget':0.0000001}
kernel_param = {'theta0':0.1}
acqui_param = {'kappa':4} 
svcBO.optimize(kernel_param=kernel_param, acqui_param=acqui_param, n_iter=100, acqui_type='ucb', n_acqui_iter=200)
print('Final Results')                             
print('SVC: %f' % svcBO.report['best']['best_val'])
plt.show()
