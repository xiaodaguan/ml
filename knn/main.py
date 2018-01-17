import kNN

group, labels = kNN.createDataSet()
print(group)
print(labels)

print(kNN.classify0([0.8, 0.9], group, labels, 3))
