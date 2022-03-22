import sklearn
from sklearn.model_selection import *
import numpy as np
from randomForest import *
from gdrakakis import *

trainFeatures_ID, trainFeatures_X, \
trainFeatures_Y, trainData_ID, \
trainData_X, trainData_Y = getDataFromTSV("dummy-dataset.txt")

def make_v_sets(trainData_X, trainData_Y):
    skf = StratifiedKFold(n_splits=3)

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    trainData_X = np.asarray(trainData_X)
    trainData_Y = np.asarray(trainData_Y)

    for tr, te in skf.split(trainData_X, trainData_Y):
        train_x.append(trainData_X[tr])
        train_y.append(trainData_Y[tr])
        test_x.append(trainData_X[te])
        test_y.append(trainData_Y[te])
    print("Train x:", train_x, '\nTrain y:', train_y)
    print("Test x:", test_x, '\nTest y:', test_y)

make_v_sets(trainData_X, trainData_Y)