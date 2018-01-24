import trees

mat, labels = trees.createDataSet()
ent = trees.calcShannonEnt(mat)
print(ent)
mat[0][-1] = 'maybe'
ent = trees.calcShannonEnt(mat)
print(ent)

# mat, labels = trees.createDataSet()
# print(mat)
# split1 = trees.splitDataSet(mat, 0, 1)
# print(split1)
# split2 = trees.splitDataSet(mat, 0, 0)
# print(split2)




# mat, labels = trees.createDataSet()
# feat = trees.chooseBestFeatureToSplit(mat)
# print(feat)
