'''
python script for optimization of parameters of the Random Forest classifier
with Bayesian Optimization
using Pure Python implementation of bayesian global optimization with gaussian processes
by fmfn
see: https://github.com/fmfn/BayesianOptimization/

prerequisities:
- pip install bayesian-optimization

'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization

import numpy as np
import pandas as pd

#
# Bounded region of parameter space
#

parameterDict = { 'n_estimators': (100, 1200),
            'max_depth': (5, 30),
            'min_samples_split': (2, 100),
            'min_samples_leaf': (1, 10)
            }


# bayesian configuration

init_points = 5
n_iter = 20


# -------------------------------------------------------------------------------- #


def function_to_be_optimized(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    '''
    wrapper for the ML function. It takes all parameters to be optimized
    and returns the score to be maximized/minimized
    in case of this classifier it is the average AUROC  cross_val_score from RandomForestClassifier
    '''

    n_estimators = int(round(n_estimators, 0))
    max_depth = int(round(max_depth))
    min_samples_split = int(round(min_samples_split))
    min_samples_leaf = int(round(min_samples_leaf))

    # print n_estimators, max_depth, min_samples_split, min_samples_leaf

    return cross_val_score(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, criterion='gini', max_features='auto'),
        X=features, y=target, scoring='roc_auc', cv=3).mean()


# -------------------------------------------------------------------------------- #

# read sample data from file - if run from the command line
# input_table = pd.read_csv("nr-ahr-lite.csv", delimiter=",")

features = input_table.iloc[:,:-1]		# data - all but the last column
target = input_table.iloc[:,-1]		# class - the last column


optimizer = BayesianOptimization(f = function_to_be_optimized, pbounds=parameterDict)

optimizer.maximize(init_points=init_points, n_iter=n_iter)

# dict with best params
bestParams = optimizer.max['params']

# best CV value achieved
bestTarget = optimizer.max['target']

# get the integers, as RF require ints
bestParamsInt = { x: int(round(y))  for x,y in bestParams.items()}

print "Best params:", bestParamsInt
print "Best target value:", bestTarget

# output models
output_model = RandomForestClassifier(n_estimators = bestParamsInt['n_estimators'], max_depth = bestParamsInt['max_depth'],
    min_samples_split = bestParamsInt['min_samples_split'], min_samples_leaf = bestParamsInt['min_samples_leaf'], criterion='gini', max_features='auto')

# Fit the model using the features and labels
output_model.fit(features, target)
