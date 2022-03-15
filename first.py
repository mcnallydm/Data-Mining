from random import random
import sklearn
import math

def main():
    while(True):
        #print("Enter the number of faces on the die: ")
        sides = input("Enter the number of faces on the die: ")
        print("\n" + roll_die(sides))
        break
    return 0

def roll_die(num_faces):
    return (random()%num_faces)+1