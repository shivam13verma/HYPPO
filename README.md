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
* http://people.seas.harvard.edu/~jsnoek/nips2013transfer.pdf
* http://www.cs.toronto.edu/~zemel/documents/bayesopt-warping.pdf

To learn more about GPs, watch the amazing lectures of Prof. Nando de Freitas on the topic.
[GitHub](http://github.com)


