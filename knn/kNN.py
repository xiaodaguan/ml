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
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 扩展成 4x1 矩阵，和dataset减法
    sqDiffMat = diffMat ** 2 # 平方
    sqDistances = sqDiffMat.sum(axis=1) # 求和
    distances = sqDistances ** 0.5 # 开方，得到距离
    sortedDistIndices = distances.argsort() # 距离排序
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 计数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # 按count 从大到小排序
    return sortedClassCount[0][0]
