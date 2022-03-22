import sklearn
from sklearn.ensemble import *
import numpy as np

def rand_forests(trainData_X, trainData_Y):
    rf = RandomForestClassifier(
        n_estimators = 100,
        criterion = "gini",
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        #min_weight_fraction_leaf = 0.0,
        max_features = "auto",
        #max_leaf_nodes = None,
        bootstrap = True,
        #oob_score = False,
        #random_state = None,
        #verbose = 0,
        #warm_start= False,
        #class_weight = None
    )