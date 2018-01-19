# encoding=utf-8
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


'''
inX: 输入的每一个点
dataSet: 数据集
labels: 标签集
k: 选择近邻数目
'''


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 扩展成 4x1 矩阵，和dataset减法
    sqDiffMat = diffMat ** 2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 求和
    distances = sqDistances ** 0.5  # 开方，得到距离
    sortedDistIndices = distances.argsort()  # 距离排序
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 计数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 按count 从大到小排序
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    mat = zeros((numberOfLines, 3))
    labels = []
    labelSet = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        mat[index, :] = listFromLine[0:3]
        if listFromLine[-1] not in labelSet:
            labelSet.append(listFromLine[-1])
        labels.append(labelSet.index(listFromLine[-1]))
        index = index + 1
    return mat, labels


'''
归一化
'''


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10  # 测试集占比
    mat, labels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(mat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], labels[numTestVecs:m], 3)
        print("the classifier says: %d, the real answer is: %d" % (classifierResult, labels[i]))
        if classifierResult != labels[i]: errCount += 1.0
    print("the total err rate is: %f" % (errCount / float(numTestVecs)))


def img2Vector(filename):
    returnVec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            try:
                returnVec[0, 32 * i + j] = int(lineStr[j])
            except:
                print(lineStr)
    return returnVec


from os import listdir


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)  # 文件数 = 训练集大小
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

        trainingMat[i, :] = img2Vector("trainingDigits/%s" % fileNameStr)
    testFileList = listdir('testDigits')
    errCount = 0.0
    t = len(testFileList)
    for i in range(t):
        fileNameStr = testFileList[i]
        label = int(fileNameStr.split('_')[0])
        testVec = img2Vector("testDigits/%s" % fileNameStr)
        result = classify0(testVec, trainingMat, hwLabels, 3)
        if (result != label): errCount += 1.0
    print("errCount = %d, errRate= %f" % (errCount, errCount / float(t)))
