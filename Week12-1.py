from sklearn.model_selection import StratifiedKFold
import numpy as np

# sklearn and np arrays 

x = np.array( [ [-1,7], [1,5], [1,2], [-2,0], \
    [2,1], [-2,0], [-1,1], [1,1], [-2,2], \
    [2,7], [-2,1], [-2,7] ] )
    
y = np.array( [1,1,1,1,2,1,1,2,1,2,2,2] )

skf = StratifiedKFold(n_splits = 3)
for train, test in skf.split(x,y):
    print (train,test) 