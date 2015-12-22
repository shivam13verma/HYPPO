'''
Created on Dec 17, 2015

@author: adarsh
'''

from model import Model
from sklearn.datasets import load_digits

from sklearn.svm import SVC
import numpy as np

digits = load_digits()
data = digits.data
target = digits.target
# print target

# digits = np.genfromtxt('mnist_background_images/mnist_background_images_train.amat', delimiter='   ')
# target = digits[:, -1]
# digits = digits[:, :-1]
# 
# for idx, image in enumerate(digits):
#     digits[idx] = image.reshape(1, -1)
# 
# print target

t = 10
logit_param_grid = { "kernel":['poly'],
                    "C": 2.0 ** np.arange(-20, 10)}

logit = Model(SVC(), logit_param_grid,
              data,
              target,
              cv=10, scorer='accuracy', search_type='grid')
print 'Running Grid search'
logit.run_grid_search()
print logit.best_params()
print logit.report()
print logit.get_grid()
