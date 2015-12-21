'''
Created on Jul 15, 2015

@author: adarsh
'''
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import sklearn.metrics as metrics
from sklearn.cross_validation import train_test_split


class Model(object):
    # TODO: time everything.

    def __init__(self, clf, param_grid, data, target, cv=10,
                 scorer="roc_auc", search_type='random'):
        '''
        Pass in a scikit learn classifier and fit()
        of this olass runs gridsearch over auc.
        :param clf: scikit learn classifier
        :param param_grid : parameter grid that fits into the clf.
            Look at GridSearchCV
        :param cv: number of folds for cross val.
        :param scorer: a list of scoring functions.
        :param data : data set we want to predict on
        :param target : Truth values for what we wish to predict.
        :param search_type: 'grid' or 'random'
        '''
        self.clf = clf
        self.data = data
        self.target = target

        self.train, self.test, self.train_labels, self.test_labels = \
            train_test_split(self.data, self.target,
                             train_size=0.7, random_state=10)
        self.param_grid = param_grid
        self.cv = cv
        self.scorer = scorer
        if search_type == 'grid':
            self.grid_search = self.__create_grid_search__()
        elif search_type == 'random':
            self.grid_search = self.__create_randomized_search__()

    def __create_grid_search__(self):
        gs = GridSearchCV(estimator=self.clf, param_grid=self.param_grid,
                          cv=self.cv, scoring=self.scorer, n_jobs=-1,
                          verbose=1)
        return gs

    def __create_randomized_search__(self):
        gs = RandomizedSearchCV(estimator=self.clf,
                                param_distributions=self.param_grid,
                                cv=self.cv, scoring=self.scorer, n_jobs=-1,
                                n_iter=100,
                                verbose=1)
        return gs

    def run_grid_search(self):
        '''
        runs grid search on the params. this obj was initialized with.
        '''
        self.grid_search.fit(self.train, self.train_labels)

    def predict(self):
        predictions = self.grid_search.predict(self.test)
        return predictions

    def best_params(self):
        return self.grid_search.best_params_

    def report(self):
        truth, predicted = self.test_labels, self.predict()
        # scorer = getattr(metrics, self.scorer)

        report = metrics.accuracy_score(truth, predicted)
        return report

    def get_best_training_score(self):
        return self.grid_search.best_score_

    def get_best_estimator(self):
        return self.grid_search.best_estimator_
    def get_grid(self):
        return self.grid_search.grid_scores_
    def predict_proba(self):
        best_estimator = self.grid_search.best_estimator_
        predictions = best_estimator.predict_proba(self.test)
        return predictions
