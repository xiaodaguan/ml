


list = [1,2,3,4,5]

print(list[:3])
print(list[0:3])
print(list[1:3])

reducedVec = list[:3]
reducedVec.extend(list[4:])
print(reducedVec)