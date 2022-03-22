import sklearn
import numpy as np
#from makeValidationSets import *

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
    
    trainData_ID = []
    trainData_X = [] 
    trainData_Y = []
    
    data = open(file)

    currentLine = data.readline()
    if not currentLine:
        return 0 # exits if there is no line read
    else:
        currentLine = currentLine.strip()
        currentLine = currentLine.split("\t") # delimiter. commonly "," or "\t"
        trainFeatures_ID = currentLine.pop(0) # pops first element in the list
        trainFeatures_Y = currentLine.pop()   # pops last element in the list
        trainFeatures_X = currentLine         # the remaining elements are kept
        
        while (True):
            currentLine = data.readline()
            print (currentLine)
            if not currentLine:
                break
            else:
                currentLine = currentLine.strip()
                currentLine = currentLine.split("\t") 
                trainData_ID.append(currentLine.pop(0))
                trainData_Y.append(currentLine.pop())
                trainData_X.append(currentLine)
    return trainFeatures_ID, trainFeatures_X, trainFeatures_Y, \
            trainData_ID, trainData_X, trainData_Y

"""
    *prints data read from file*
    - file specs are tailored to the output (return) 
      of the function getDataFromTSV (see above)
"""
def printInputData (file):
    trainFeatures_ID, trainFeatures_X, \
    trainFeatures_Y, trainData_ID, \
    trainData_X, trainData_Y = getDataFromTSV(file)
    
    print ("trainFeatures_ID: ", trainFeatures_ID, "\n", \
          "trainFeatures_X: ", trainFeatures_X, "\n", \
          "trainFeatures_Y: ", trainFeatures_Y, "\n", \
          "trainData_ID: ", trainData_ID, "\n", \
          "trainData_X: ", trainData_X, "\n", \
          "trainData_Y: ", trainData_Y)
    return 0
    
"""
    RUN area
"""
printInputData (inputFile)