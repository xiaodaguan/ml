import kNN
import matplotlib
import numpy
from numpy import *
import matplotlib.pyplot as plt

# group, labels = kNN.createDataSet()
# print(group)
# print(labels)

# print(kNN.classify0([0.8, 0.9], group, labels, 3))


group, labels = kNN.file2matrix('datingTestSet.txt')
# print(group)
# print(labels)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(group[:, 1], group[:, 2], 15.0 * array(labels), 15.0 * array(labels))
# plt.show()


normMat, ranges, minVals = kNN.autoNorm(group)
