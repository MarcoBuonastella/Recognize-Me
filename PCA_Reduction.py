import matplotlib.pyplot as plt
import math
import numpy as np
from numpy import linalg as linalg
import pandas as pd
import os as os
from os import listdir
from os.path import isfile, join
import random
import statistics
from scipy.sparse.linalg import eigs
from sklearn.neighbors import KNeighborsClassifier

X_TrainFileName = os.path.abspath('') + "\\" + "X_train.csv"
X_TestFileName = os.path.abspath('') + "\\" + "X_test.csv"
Y_TrainFileName = os.path.abspath('') + "\\" + "Y_train.csv"
Y_TestFileName = os.path.abspath('') + "\\" + "Y_test.csv"

trainLabels = pd.read_csv(Y_TrainFileName, header=None)
testLabels =  pd.read_csv(Y_TestFileName, header=None)

trainData = np.genfromtxt(X_TrainFileName,delimiter=',')
print(len(trainData),len(trainData[0]))

#partA Visualizing Eigenfaces
means = np.mean(trainData, axis=0)
trainData = trainData - means
covMatrix = np.cov(trainData,rowvar=False)


test = np.array([[.69,.49],[-1.31,-1.21],[.39,.99],[.09,.29],[1.29,1.09],
[.49,.79],[.19,-.31],[-.81,-.81],[-.31,-.31],[-.71,-1.01]])

testCov = np.cov(test,rowvar=False)


vals, vecs = eigs(covMatrix, k=50, which='LR')
vecs = vecs.transpose()

eigenVecValuePairs = []

for i in range(0,len(vecs)):
	vec = vecs[i]
	val = vals[i]
	eigenVecValuePairs.append([vec,val])

eigenVecValuePairs.sort(reverse = True , key=lambda x: x[1])

#top 50 eigenvectors
eigenVecValuePairsTop50 = eigenVecValuePairs[:50]

#display top 10 as image
eigenVecValuePairsTop10 = eigenVecValuePairs[:10]

#transposed eigenvectors
eigenVecTop10Transpose = []
eigenVecTop50Transpose = []

for i in range(0,len(eigenVecValuePairsTop50)):
	eigenVecTop50Transpose.append(eigenVecValuePairsTop50[i][0])

for i in range(0,len(eigenVecValuePairsTop10)):
	eigenVecTop10Transpose.append(eigenVecValuePairsTop10[i][0])


trainDataTranspose = np.transpose(trainData)
eigenVecTop10Transpose = (np.array(eigenVecTop10Transpose)).real
eigenVecTop50Transpose = (np.array(eigenVecTop50Transpose)).real

#print(len(eigenVecTop10Transpose), len(eigenVecTop10Transpose[0]))
#print(eigenVecTop10Transpose[0])
#plt.imshow(np.reshape(eigenVecTop10Transpose[0,:],(92,112)),cmap="gray")
#plt.show()

#project data onto top 50 vectors

#newDataSet = np.dot(eigenVecTop50Transpose,trainDataTranspose)
#newDataSet = newDataSet.transpose()

print(len(eigenVecTop50Transpose), len(eigenVecTop50Transpose[0]))
print(len(trainDataTranspose), len(trainDataTranspose[0]))

#U = []
#print(len(np.transpose(trainData[0])))
#print(np.dot(np.transpose(trainData[0]),eigenVecTop50Transpose[0]))
#for i in range(0,len(eigenVecTop50Transpose)):
	#ui = np.matmul(trainDataTranspose[0],eigenVecTop50Transpose[i])
	#U.append(ui)

#print(U)


#1st ----------------------
plt.figure(1)
originalImage = trainData[0]
plt.imshow(np.reshape(originalImage,(92,112)),cmap="gray")

plt.figure(2)
U = []

for i in range(0,len(eigenVecTop50Transpose)):
	ui = np.dot(np.transpose(trainData[0]),eigenVecTop50Transpose[i])
	U.append(ui)

newImage = np.dot(U,eigenVecTop50Transpose)
plt.imshow(np.reshape(newImage,(92,112)),cmap="gray")

#5th ---------------------
plt.figure(3)
originalImage = trainData[4]
plt.imshow(np.reshape(originalImage,(92,112)),cmap="gray")


plt.figure(4)
U = []

for i in range(0,len(eigenVecTop50Transpose)):
	ui = np.dot(np.transpose(trainData[4]),eigenVecTop50Transpose[i])
	U.append(ui)

newImage = np.dot(U,eigenVecTop50Transpose)
plt.imshow(np.reshape(newImage,(92,112)),cmap="gray")


#20th ---------------------
plt.figure(5)
originalImage = trainData[19]
plt.imshow(np.reshape(originalImage,(92,112)),cmap="gray")


plt.figure(6)
U = []

for i in range(0,len(eigenVecTop50Transpose)):
	ui = np.dot(np.transpose(trainData[19]),eigenVecTop50Transpose[i])
	U.append(ui)

newImage = np.dot(U,eigenVecTop50Transpose)
plt.imshow(np.reshape(newImage,(92,112)),cmap="gray")


#30th -------------------
plt.figure(7)
originalImage = trainData[29]
plt.imshow(np.reshape(originalImage,(92,112)),cmap="gray")


plt.figure(8)
U = []

for i in range(0,len(eigenVecTop50Transpose)):
	ui = np.dot(np.transpose(trainData[29]),eigenVecTop50Transpose[i])
	U.append(ui)

newImage = np.dot(U,eigenVecTop50Transpose)
plt.imshow(np.reshape(newImage,(92,112)),cmap="gray")


#40th
plt.figure(9)
originalImage = trainData[39]
plt.imshow(np.reshape(originalImage,(92,112)),cmap="gray")


plt.figure(10)
U = []

for i in range(0,len(eigenVecTop50Transpose)):
	ui = np.dot(np.transpose(trainData[39]),eigenVecTop50Transpose[i])
	U.append(ui)

newImage = np.dot(U,eigenVecTop50Transpose)
plt.imshow(np.reshape(newImage,(92,112)),cmap="gray")

#plt.show()

#part 3
trainLabels = testLabels = np.genfromtxt(Y_TrainFileName,delimiter=',')
testLabels = np.genfromtxt(Y_TestFileName,delimiter=',')
testData = np.genfromtxt(X_TestFileName,delimiter=',')
testData = testData - means

newTestImages = []
newTrainImages = []

for j in range(0,len(testData)):
	U = []

	for i in range(0,len(eigenVecTop50Transpose)):
		ui = np.dot(np.transpose(testData[j]),eigenVecTop50Transpose[i])
		U.append(ui)

	newImage = np.dot(U,eigenVecTop50Transpose)
	newTestImages.append(newImage)


for j in range(0,len(trainData)):
	U = []

	for i in range(0,len(eigenVecTop50Transpose)):
		ui = np.dot(np.transpose(trainData[j]),eigenVecTop50Transpose[i])
		U.append(ui)
	
	newImage = np.dot(U,eigenVecTop50Transpose)
	newTrainImages.append(newImage)


KNN = KNeighborsClassifier(n_neighbors=1)
KNN.fit(newTrainImages,trainLabels)
print(KNN.score(newTestImages,testLabels))

 
 







