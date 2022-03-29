from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

inputFile = "dummy-dataset.txt"

"""
    *reads data from file*
    - row by row
    - assumes that:
        - the first row contains variable names
        - the first column contains IDs
        - the last column contains the class         
"""
def getDataFromTSV (file):
    
    inputFileData_ID = []
    inputFileData_X = [] 
    inputFileData_Y = []
    
    data = open(file)

    currentLine = data.readline()
    if not currentLine:
        return 0 # exits if there is no line read
    else:
        currentLine = currentLine.strip()
        currentLine = currentLine.split("\t") # delimiter. commonly "," or "\t"
        inputFileFeatures_ID = currentLine.pop(0) # pops first element in the list
        inputFileFeatures_Y = currentLine.pop()   # pops last element in the list
        inputFileFeatures_X = currentLine         # the remaining elements are kept
        
        while (True):
            currentLine = data.readline()
            print (currentLine)
            if not currentLine:
                break
            else:
                currentLine = currentLine.strip()
                currentLine = currentLine.split("\t") 
                inputFileData_ID.append(currentLine.pop(0))
                inputFileData_Y.append(currentLine.pop())
                inputFileData_X.append(currentLine)
    return inputFileFeatures_ID, inputFileFeatures_X, inputFileFeatures_Y, \
            inputFileData_ID, inputFileData_X, inputFileData_Y

"""
    *prints data read from file*
    - file specs are tailored to the output (return) 
      of the function getDataFromTSV (see above)
"""
def printInputData (file):
    inputFileFeatures_ID, inputFileFeatures_X, \
    inputFileFeatures_Y, inputFileData_ID, \
    inputFileData_X, inputFileData_Y = getDataFromTSV(file)
    
    print ("inputFileFeatures_ID: ", inputFileFeatures_ID, "\n", \
          "inputFileFeatures_X: ", inputFileFeatures_X, "\n", \
          "inputFileFeatures_Y: ", inputFileFeatures_Y, "\n", \
          "inputFileData_ID: ", inputFileData_ID, "\n", \
          "inputFileData_X: ", inputFileData_X, "\n", \
          "inputFileData_Y: ", inputFileData_Y)
    return 0

"""
    missing description
"""
def itcRandomForest (inputFileData_X, inputFileData_Y): ## pass RF parameters here if you want 
    itcRF = RandomForestClassifier(
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
                #warm_start = False, 
                #class_weight = None
                )
    itcRF.fit (inputFileData_X, inputFileData_Y) 	
    return itcRF


"""
    RUN area
"""

# get data from input file using getDataFromTSV function
inputFileFeaturesID, inputFileFeatures_X, \
inputFileFeatures_Y, inputFileData_ID, \
inputFileData_X, inputFileData_Y = getDataFromTSV(inputFile)

# initialise StratifiedKFold and arrays for our data (we will 'append' to them)
skf = StratifiedKFold(n_splits = 3)
train_x = []
train_y = []
test_x = []
test_y = []

# if need be, cast as numpy arrays
inputFileData_X = np.asarray(inputFileData_X)
inputFileData_Y = np.asarray(inputFileData_Y) 

# allocate according to indices provided
for tr, te in skf.split(inputFileData_X,inputFileData_Y):
    print ("Indices: \n", tr, "\n", te) # print to confirm correctness
    train_x.append(inputFileData_X[tr])
    train_y.append(inputFileData_Y[tr])
    test_x.append(inputFileData_X[te])
    test_y.append(inputFileData_Y[te])

# confirm it works, can comment out afterwards
print ('train x: \n', train_x, '\n train y\n', train_y)
print ('test x: \n', test_x, '\n test y\n', test_y)

# start model / prediction here
