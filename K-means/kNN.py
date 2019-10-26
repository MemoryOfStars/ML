# -*- coding: utf-8 -*-
"""
K-Means Algorithm
Created on Tue Sep 24 16:16:07 2019

@author: 75100
"""
from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group,labels
    
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                   #Feature
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  #row
    sqDiffMat = diffMat**2                     #Squre
    sqDistances = sqDiffMat.sum(axis=1)        #Sum of Squares
    distances = sqDistances**0.5               #Rooting
    sortedDistIndicies = distances.argsort()   #Sort(return index order)
    classCount = {}                            #Dict
    for i in range(k):#Get the shortest k spots
        voteIlabel = labels[sortedDistIndicies[i]]   #第i个spot的label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
        
if __name__ == '__main__':
    group,labels = createDataSet()
    print(classify0([0.9,0.9], group, labels, 2))