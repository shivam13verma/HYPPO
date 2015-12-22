'''
Created on Dec 18, 2015

@author: adarsh
'''

from hypop import BayesOptCV
from solutions_helper import *
import glob
trainfiles = glob.glob('/home/adarsh/Copy/SharedWorkspace/inf_ps_7/Data/train*.txt')
testfiles = glob.glob('/home/adarsh/Copy/SharedWorkspace/inf_ps_7/Data/test*.txt')
train_data, train_labels = transform_data(trainfiles)
train_data = encode_sentences(train_data)
train_size = 100
def struct_predict(C):
    val_score = bpo_cost_fun(train_data[
                               :train_size], train_data[-500:], train_labels[:train_size], train_labels[-500:], C)
    return val_score
    
                           
svcBO = BayesOptCV(struct_predict, param_grid={'C':{'type':'float', 'min':6.4e-05, 'max':100}}, verbose=2)

svcBO.initialize(num_init=10, init_grid={'C': [0.001, 0.1, 1, 10, 50, 80, 100, ]})
# kernel_param = {'nugget':0.0000001}
kernel_param = {'theta0':1.0}
acqui_param = {} 
svcBO.optimize(kernel_param=kernel_param, acqui_param=acqui_param, n_iter=10, n_acqui_iter=50)
print('Final Results')                             
print('StructSVM: %f' % svcBO.report['best']['best_val'])
plt.show()
