# -*- coding=utf-8 -*-
# Time: 2021/4/17 0017
# Name: kNN.py
# Tool: PyCharm
# @author: Curry
# Address: zm_seu@seu.edu.cn

# coding=utf-8
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

from pip._vendor.distlib.compat import raw_input

group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
labels = ['A', 'A', 'B', 'B']


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # shape[0]矩阵的行数,shape[1]矩阵列数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 扩展坐标维数与数据集维数相同并两矩阵整体坐标相减
    sqDiffMat = diffMat ** 2  # 乘方
    sqDistances = sqDiffMat.sum(axis=1)  # axis=1 将一个矩阵的每一行中的元素相加
    distances = sqDistances ** 0.5  # distances = ((tile(inX, (dataSetSize, 1)) - dataSet) ** 2).sum(axis=1)
    sortedDistIndex = distances.argsort()  # 返回的是数组值从小到大的索引值
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  # 计算k个元素中标签出现次数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # reverse=True反向排序
    # python2用iteritems，python3下用items()，返回可遍历的(键, 值) 元组数组
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)  # get the number of lines in the file
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    index = 0
    for line in arrayOfLines:
        line = line.strip()  # 移除字符串头尾指定的字符
        listFromLine = line.split('\t')  # 制表符分片
        returnMat[index, :] = listFromLine[0:3]  # 选前三列的数据存到矩阵中
        classLabelVector.append(int(listFromLine[-1]))  # 最后一列转成整数后存到标签向量
        index += 1
    return returnMat, classLabelVector


def setVisualization():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()


#  归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 每一列最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    print(ranges)
    normDataSet = zeros(shape(dataSet))  # 返回矩阵大小和数据矩阵一样用0填充
    m = dataSet.shape[0]  # 矩阵行数
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.90  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 6)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]: errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


def classifyPerson():
    resultList = ['不喜欢', '魅力一般的人', '极具魅力的人']
    percentTats = float(raw_input("percentgage of time spent playing video game ?"))
    ffMile = float(raw_input("frequent flier miles earned per year ?"))
    iceCream = float(raw_input("liters of ice Cream consumed per year ?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMile, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult-1])


classifyPerson()
