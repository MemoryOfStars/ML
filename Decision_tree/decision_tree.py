# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:57:19 2019
Decision Tree
(1.Calculate Shannon entropy)
(2.)
@author: 75100
"""

from math import log

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calculateShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():#keys()returns all index in dict
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:#该分类中的项数除以数据集中数据总数 即 某数据项落于此分类中的概率
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
    
if __name__ == '__main__':
    data, labels = createDataSet()
    print(calculateShannonEnt(data))