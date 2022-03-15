'''
WHAT WE NEED:
- Probability of each outcome
- Value of each outcome
- Total # of instances
- 2+ Groups
- Entropy formula
- Dataset
- A way to input/read data
'''
# TO RUN FILE USE python filename.py

import math
from cmath import log
from numpy import np

play = [9, 5]
# yes, no

outlook = [
    [4, 0], # overcast
    [3, 2], # rainy
    [2, 3]  # sunny
]

humidity = [
    [3, 4], # high
    [6, 1]  # normal
]

temperature = [
    [3, 1], # cool
    [2, 2], # mild
    [4, 2]  # hot
]

windy = [
    [6, 2], # false
    [3, 3]  # true
]

'''
def class_index(dataset):
    return len(dataset[0])-1    # returns the index of the last attribute of a datapoint (aka the class)
'''

def get_tsv_data(file):
    trainData_id = []
    trainData_x = []
    trainData_y = []
    

    data = open(file)
    current_line = data.readline()
    current_line = current_line.strip()
    current_line = current_line.split("\t")
    trainFeatures_id = current_line.pop(0)
    trainFeatures_x = current_line.pop()
    trainFeatures_y = current_line
    while(1):
        current_line = data.readline()
        if not current_line:
            break
        else:
            current_line = current_line.strip()
            current_line = current_line.split("\t")
            trainData_id.append(current_line.pop(0))
            trainData_y.append(current_line.pop())
            trainData_x.append(current_line)
    return trainFeatures_id, trainFeatures_x, trainFeatures_y, trainData_id, trainData_x, trainData_y

def my_open_tsv(file):
    master_list = []
    return master_list

def entropy(class_count):  # list of counts of each outcome
    total = 0
    total_count = sum(class_count)
    for p in class_count:
        p = p/total_count
        if p != 0:
            total += p * log(p, 2)
    total *= -1
    return total

def lin_search(item, arr):
    idx = -1
    for element in range(0, len(arr)):
        if arr[element] == item:
            idx = element
            break
    return idx

def count_classes(dataset):
    class_types = []
    class_count = []
    col = len(dataset[0])-1
    for datapoint in dataset:
        count_idx = lin_search(datapoint[col], class_types)
        if count_idx!=-1:
            class_count[count_idx] += 1
        else:
            class_types.append(datapoint[col])
            class_count.append(1)
    return class_count

def list_attribute(dataset, col):
    final_list = []
    att_types = []
    for element in dataset:
        if lin_search(element[col], att_types) == -1:
            att_types.append(element[col])
    temp = []
    for attribute in att_types:
        for datapoint in dataset:
            if datapoint[col] == attribute:
                temp.append(datapoint)
        final_list.append(count_classes(temp))
    return final_list

def info_gain(class_var, a):
    total = 0
    for val in a:
        total += (sum(val)/sum(class_var)) * entropy(val)
    ig = entropy(class_var) - total
    return ig

def process(dataset):
    num_att = len(dataset[0])-1
    att_lists = count_classes(dataset)
    for i in range(1, num_att):
        att_lists.append(list_attribute(dataset, i))


print(info_gain(play, outlook))
print(info_gain(play, temperature))
print(info_gain(play, windy))
print(info_gain(play, humidity))