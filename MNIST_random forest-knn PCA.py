import numpy as np
import pandas as pd
import seaborn as sb
sb.set_style("dark")
import matplotlib.pyplot as plt
from matplotlib import figure
from IPython.core.pylabtools import figsize

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import time

# We use this function in order to evaulate a classifier. It trains on a fraction of the data corresponding to
# aplit_ratio, and evaulates on the rest of the data

def evaluate_classifier(clf, data, target, split_ratio):
    trainX, testX, trainY, testY = train_test_split(data, target, test_size=split_ratio, random_state=0,stratify=target)
    clf.fit(trainX, trainY)
    return clf.score(testX,testY)
# read in the data

train = pd.read_csv('/Users/jiaxiaoyu/Downloads/train_mnist.csv')
print(train.shape)
#test  = pd.read_csv('../input/test.csv')
target = train["label"]
train = train.drop("label",1)

# plot some of the numbers

plt.figure(figsize(5,5))
for digit_num in range(0,64):
    plt.subplot(8,8,digit_num+1)
    grid_data = train.iloc[digit_num].values.reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")
    plt.xticks([])
    plt.yticks([])
plt.show()

# check performance of random forest classifier, as function of number of estimators
# here we only take 1000 data points to train
n_estimators_array = np.array([1,5,10,50,100,200,500])
n_samples = 10
n_grid = len(n_estimators_array)
score_array_mu =np.zeros(n_grid)
score_array_sigma = np.zeros(n_grid)
j=0
for n_estimators in n_estimators_array:
    score_array=np.zeros(n_samples)
    for i in range(0,n_samples):
        clf = RandomForestClassifier(n_estimators = n_estimators, n_jobs=1, criterion="gini")
        score_array[i] = evaluate_classifier(clf, train.iloc[0:1000], target.iloc[0:1000], 0.2)
    score_array_mu[j], score_array_sigma[j] = np.mean(score_array), np.std(score_array)
    j=j+1

# it looks like the performace saturates around 50-100 estimators

plt.figure(figsize(7,5))
plt.errorbar(n_estimators_array, score_array_mu, yerr=score_array_sigma, fmt='k.-')
plt.xscale("log")
plt.xlabel("number of estimators",size = 20)
plt.ylabel("accuracy",size = 20)
plt.xlim(0.9,600)
plt.grid(which="both")
plt.show()

#Are there any feature that are particularly important? We can check this using clf.feature_importances:
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(0,10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances

plt.figure(figsize(7,5))
plt.plot(indices[:],importances[indices[:]],'k.')
plt.yscale("log")
plt.xlabel("feature",size=20)
plt.ylabel("importance",size=20)
plt.show()


#there are no significantly important features ,let us try to decompose the data using a principal component analysis (PCA)

pca = PCA(n_components=2)
pca.fit(train)
transform = pca.transform(train)

plt.figure(figsize(6,5))
plt.scatter(transform[:,0],transform[:,1], s=20, c = target, cmap = "nipy_spectral", edgecolor = "None")
plt.colorbar()
plt.clim(0,9)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

n_components_array=([1,2,3,4,5,10,20,50,100,200,500])
vr = np.zeros(len(n_components_array))
i=0;
for n_components in n_components_array:
    pca = PCA(n_components=n_components)
    pca.fit(train)
    vr[i] = sum(pca.explained_variance_ratio_)
    i=i+1

plt.figure(figsize(8,5))
plt.plot(n_components_array,vr,'k.-')
plt.xscale("log")
plt.ylim(9e-2,1.1)
plt.yticks(np.linspace(0.2,1.0,9))
plt.xlim(0.9)
plt.grid(which="both")
plt.xlabel("number of PCA components",size=20)
plt.ylabel("variance ratio",size=20)
plt.show()

#KNN
clf = KNeighborsClassifier()
n_components_array=([1,2,3,4,5,10,20,50,100,200,500])
score_array = np.zeros(len(n_components_array))
i=0

for n_components in n_components_array:
    pca = PCA(n_components=n_components)
    pca.fit(train)
    transform = pca.transform(train.iloc[0:1000])
    score_array[i] = evaluate_classifier(clf, transform, target.iloc[0:1000], 0.2)
    i=i+1

plt.figure(figsize(8,5))
plt.plot(n_components_array,score_array,'k.-')
plt.xscale('log')
plt.xlabel("number of PCA components", size=20)
plt.ylabel("accuracy", size=20)
plt.grid(which="both")
plt.show()

# PCA + kNN
start1=time.time()
start2=time.process_time()
PCA = PCA(n_components=50)
PCA.fit(train)
transform_train = pca.transform(train)
KNN = KNeighborsClassifier()
print('KNN: '+ str((evaluate_classifier(KNN, transform_train, target, 0.2)*100))+'%')
end1=time.time()
end2=time.process_time()

#random forest
start3=time.time()
start4=time.process_time()
RF = RandomForestClassifier(n_estimators = 100, n_jobs=1, criterion="gini")
print('RF: '+ str((evaluate_classifier(RF, train, target, 0.2)*100))+'%')
end3=time.time()
end4=time.process_time()
clock1 = end1 - start1
cpu1 = end2-start2
clock2 = end3 - start3
cpu2 = end4-start4
print("KNN wall-clock time is" + str(clock1) + "s")
print("KNN run-time is" + str(cpu1) + "s")
print("RF wall-clock time is" + str(clock2) + "s")
print("RF run-time is" + str(cpu2) + "s")


