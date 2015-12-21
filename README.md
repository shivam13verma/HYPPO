# HYPPO
HYPer-Parameter Optimization: Bayesian optimization using Gaussian processes for scikit-learn.

This is a library for doing hyperparameter tuning based on in Machine Learning models using the popular library scikit-learn.

Current features:
* Does sequential Bayesian Optimization using GPs
* Easy to use and implement for current scikit-learn users
* Plots GPs and acquisition functions
* Acquisition functions implemented include EI, PI, GP-UCB
* Uses scikit-learn's Gaussian Processes module (which does not include Matern kernel as of now)

Future additions:
* Parallel Bayesian Optimization using slice sampling
* Custom Gaussian Processes (see gaussian_process folder) module including Matern32/52 kernels
* Implementation of Tree Parzen Estimators
* Multi-task Bayesian Optimization, input-warping

References:
* http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
* http://arxiv.org/pdf/1012.2599.pdf
* http://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/Ryan_adams_140814_bayesopt_ncap.pdf
* http://people.seas.harvard.edu/~jsnoek/nips2013transfer.pdf
* http://www.cs.toronto.edu/~zemel/documents/bayesopt-warping.pdf

To learn more about GPs, read the references above, or watch the amazing lectures from Prof. Nando de Freitas [on](https://www.youtube.com/watch?v=4vGiHC35j9s) [the](https://www.youtube.com/watch?v=MfHKW5z-OOA) [topic.](https://www.youtube.com/watch?v=vz3D36VXefI)

Authors (in alphabetic order):
Adarsh Jois, Shivam Verma (New York University)
