from math import log

'''
对全部标签计算信息熵
对于一个数据集
表示数据集的不确定性大小
'''


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}  # 全部label的频次统计
    # 计数 {'label1':count1, 'label2':count2, ...}
    for featVec in dataSet:  # 每一个向量
        currentLabel = featVec[-1]  # label
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:  # 每一个label
        prob = float(labelCounts[key]) / numEntries  # 概率: p{xi} = label频次/元素总数
        shannonEnt -= prob * log(prob, 2)  # 信息熵: 所有类别所有可能值包含的信息期望值 H = sum( -p(xi) log2 p(xi)  )
    return shannonEnt


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


'''
划分数据集: 去掉第axis列，将第axis列值为value的全部向量抽出来
dataSet:待划分的数据集
axis:划分数据集的特征(哪一列)
value:需要返回的特征的值
'''


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


'''
选择最佳划分数据集方式
针对一个数据集
用每个特征切分数据集，分别计算数据集的熵
得出熵差值最大的特征
'''


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 一列
        uniqueVals = set(featList)  # 去重
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 划分数据集，该特征有几个值就划分几次
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet) # 加权求和，几个value的权值和为1
        infoGain = baseEntropy - newEntropy # 信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
