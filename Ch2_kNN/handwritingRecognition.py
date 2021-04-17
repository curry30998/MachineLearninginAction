# -*- coding=utf-8 -*-
# Time: 2021/4/17 0017
# Name: handwritingRecognition.py
# Tool: PyCharm
# @author: Curry
# Address: zm_seu@seu.edu.cn

from os import listdir
from numpy import *
from Ch2_kNN.kNN import classify0


# 32*32的二进制图像矩阵转化成1*1024的向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    file = open(filename)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            returnVect[0, i * 32 + j] = line[j]
    return returnVect


# 手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # listdir()以数组形式返回指定文件夹中所有文件的名称
    m = len(trainingFileList)  # 文件个数
    trainingMat = zeros((m, 1024))  # 创建m行，1024列的零矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 读取第i个文件的名称
        fileStr = fileNameStr.split('.')[0]
        # split()函数将去除文件名中的'.'，并返回分割后的字符串列表，索引0将读取第一个元素，根据源文件的命名规则，返回值是0_0,0_89,1_25等形式的字符串
        classNumStr = int(fileStr.split('_')[0])  # 分割并读取第一个元素，转化为整型，返回值是0,1,2,3...
        hwLabels.append(classNumStr)  # 将最终得到的数值作为标签存储到列表中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
        # 将指定文件转换为测试向量存储到trainingMat的第i行中
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)

        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  # 对测试数据进行分类
        print('the classifier came back with:%d, the real answer is: %d' % (classifierResult, classNumStr))
        if classifierResult != classNumStr: errorCount += 1  # 当测试数据结果错误时，errorCount+1
    print('\nthe total number of error is : %d' % errorCount)
    print('\nthe total rate of error is : %f' % (errorCount / float(mTest)))


handwritingClassTest()
