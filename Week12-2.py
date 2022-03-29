from sklearn.model_selection import StratifiedKFold
import numpy as np

x = np.array( [ [-1,7], [1,5], [1,2], [-2,0], \
    [2,1], [-2,0], [-1,1], [1,1], [-2,2], \
    [2,7], [-2,1], [-2,7] ] )
    
y = np.array( [1,1,1,1,2,1,1,2,1,2,2,2] )

skf = StratifiedKFold(n_splits = 3)

# 3 splits for x (input variables) and y (output/class)

# training and testing are located using indices 

#[ 3  5  6  8  9 10 11] [0 1 2 4 7]
#[ 0  1  2  4  6  7  8 11] [ 3  5  9 10]
#[ 0  1  2  3  4  5  7  9 10] [ 6  8 11]

# need to get separate lists/arrays for X and Y 
# for both training and testing 

train_x = [] 
train_y = [] 
test_x = []
test_y = []

for tr, te in skf.split(x,y):
    #print(tr, te)
    train_x.append(x[tr])
    train_y.append(y[tr])
    test_x.append(x[te])
    test_y.append(y[te])

print ('\ntrain x: \n', train_x, '\n\n train y:\n', train_y)
print ('\ntest x: \n', test_x, '\n\n test y:\n', test_y)