from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn
import numpy as np

inputFile = "dummy-dataset.txt"

"""
    function description
"""
def getDataFromTSV (file):
    
    inputFileData_ID = []
    inputFileData_X = [] 
    inputFileData_Y = []
    
    data = open(file)

    currentLine = data.readline()
    if not currentLine:
        return 0
    else:
        currentLine = currentLine.strip()
        currentLine = currentLine.split("\t") 
        inputFileFeatures_ID = currentLine.pop(0)
        inputFileFeatures_Y = currentLine.pop()
        inputFileFeatures_X = currentLine
        
        while (True):
            currentLine = data.readline()
            print (currentLine)
            if not currentLine:
                break
            else:
                currentLine = currentLine.strip()
                currentLine = currentLine.split("\t") # "," or "\t"
                inputFileData_ID.append(currentLine.pop(0))
                inputFileData_Y.append(currentLine.pop())
                inputFileData_X.append(currentLine)
    return inputFileFeatures_ID, inputFileFeatures_X, inputFileFeatures_Y, \
            inputFileData_ID, inputFileData_X, inputFileData_Y

"""
    function description
"""
def itcRandomForest (inputFileData_X, inputFileData_Y): ## pass RF parameters here 
    itcRF = RandomForestClassifier(
                n_estimators = 100, 
                criterion = "gini", 
                max_depth = None, 
                min_samples_split = 2, 
                min_samples_leaf = 1, 
                min_weight_fraction_leaf = 0.0, 
                max_features = "auto", 
                max_leaf_nodes = None,  
                bootstrap = True, 
                oob_score = False, 
                random_state = None, 
                verbose = 0, 
                warm_start = False, 
                class_weight = None
                )
    itcRF.fit (inputFileData_X, inputFileData_Y) 
    return itcRF

"""
    function description
"""   
def printInputData (inputFile):
    inputFileFeaturesID, inputFileFeatures_X, \
    inputFileFeatures_Y, inputFileData_ID, \
    inputFileData_X, inputFileData_Y = getDataFromTSV(inputFile)
    
    print ("inputFileFeaturesID: ", inputFileFeaturesID, "\n", \
          "inputFileFeatures_X: ", inputFileFeatures_X, "\n", \
          "inputFileFeatures_Y: ", inputFileFeatures_Y, "\n", \
          "inputFileData_ID: ", inputFileData_ID, "\n", \
          "inputFileData_X: ", inputFileData_X, "\n", \
          "inputFileData_Y: ", inputFileData_Y)
    return 0

"""
    Classification performance
"""
def stats_classification(Y, predY, iteration):
    Accuracy = sklearn.metrics.accuracy_score(Y, predY)                   
    Precision = sklearn.metrics.precision_score(Y, predY, pos_label=None, average = 'weighted')  
    Recall = sklearn.metrics.recall_score(Y, predY, pos_label=None, average = 'weighted')        
    F1_score = sklearn.metrics.f1_score(Y, predY, pos_label=None, average = 'weighted')      

    cm = confusion_matrix(Y, predY, labels = list(set(Y)))

    # can call plot function here if we want 
    plot_confusion_matrix(cm, list(set(Y)), title='Confusion matrix')

    figfile = "myConfMat"+str(iteration)+".png"
    plt.savefig(figfile, dpi=300, format='png')
    plt.show()
    plt.close()

    return round(Accuracy,2), round(Precision,2), round(Recall,2), round(F1_score,2), figfile 
    
"""
    Matplotlib default Confusion Matrix
"""
def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.tight_layout()

"""
    RUN Area
""" 
inputFileFeaturesID, inputFileFeatures_X, \
inputFileFeatures_Y, inputFileData_ID, \
inputFileData_X, inputFileData_Y = getDataFromTSV(inputFile)

skf = StratifiedKFold(n_splits = 3)

train_x = []
train_y = []
test_x = []
test_y = []

inputFileData_X = np.asarray(inputFileData_X) # x 
inputFileData_Y = np.asarray(inputFileData_Y) # y

for tr, te in skf.split(inputFileData_X,inputFileData_Y):
    train_x.append(inputFileData_X[tr])
    train_y.append(inputFileData_Y[tr])
    test_x.append(inputFileData_X[te])
    test_y.append(inputFileData_Y[te])

# if it works, we can comment out
print ('train x: ', train_x, '\n train y', train_y)
print ('test x: ', test_x, '\n test y', test_y)

# initialise prediction list
predY = []

for i in range (0,3): # number of folds
    RFmodel = itcRandomForest(train_x[i], train_y[i]) # modify appropriately
    predY.append(RFmodel.predict(test_x[i]))

    Accuracy, Precision, Recall, \
    F1, figfile =  stats_classification(test_y[i], RFmodel.predict(test_x[i]), i)

    print ("Accuracy:", Accuracy)
    print ("Precision:", Precision)
    print ("Recall:", Recall)
    print ("F1:", F1)

# Note that these metrics are calculated per fold 

