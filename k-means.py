import time
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import homogeneity_score, completeness_score, \
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

pd.set_option('max_colwidth',1000000)
pd.set_option('display.width',2000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df=pd.read_csv('/Users/jiaxiaoyu/Downloads/train.csv')
df=df.drop(['rn'],axis=1)
#print(df.info())
Labels = df['activity']
Data=df.drop(['activity'],axis=1)
print('data shape:' +str(Data.shape))
#print(data.info())
#print(label)
Labels_keys = Labels.unique().tolist()
Labels = np.array(Labels)
print('Activity labels: ' + str(Labels_keys))
Temp = pd.DataFrame(Data.isnull().sum())
Temp.columns = ['Sum']
print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['Sum'] > 0])))
scaler = preprocessing.StandardScaler()
Data = scaler.fit_transform(Data)
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(Data)
    inertias.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()
#Looks like the best value ("elbow" of the line) for k is 2 (two clusters)
def k_means(n_clust, data_frame, true_labels):
    k_means = KMeans(n_clusters=n_clust, random_state=123, n_init=30)
    k_means.fit(data_frame)
    c_labels = k_means.labels_
    df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    y_clust = k_means.predict(data_frame)
    display(ct)
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (k_means.inertia_,
             homogeneity_score(true_labels, y_clust),
             completeness_score(true_labels, y_clust),
             v_measure_score(true_labels, y_clust),
             adjusted_rand_score(true_labels, y_clust),
             adjusted_mutual_info_score(true_labels, y_clust,average_method='max'),
             silhouette_score(data_frame, y_clust, metric='euclidean')))
start1=time.time()
start2 = time.process_time()
k_means(n_clust=2, data_frame=Data, true_labels=Labels)
end1 = time.time()
end2 = time.process_time()
clock = end1 - start1
cpu = end2-start2
print("wall-clock time is" + str(clock) + "s")
print("run-time is" + str(cpu) + "s")

k_means(n_clust=6, data_frame=Data, true_labels=Labels)

#change labels into binary: 0 - not moving, 1 - moving
Labels_binary = Labels.copy()
for i in range(len(Labels_binary)):
    if (Labels_binary[i] == 'STANDING' or Labels_binary[i] == 'SITTING' or Labels_binary[i] == 'LAYING'):
        Labels_binary[i] = 0
    else:
        Labels_binary[i] = 1
Labels_binary = np.array(Labels_binary.astype(int))
k_means(n_clust=2, data_frame=Data, true_labels=Labels_binary)

#PCA

#check for optimal number of features
pca = PCA(random_state=123)
pca.fit(Data)
features = range(pca.n_components_)

plt.figure(figsize=(8,4))
plt.bar(features[:15], pca.explained_variance_[:15], color='lightskyblue')
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.xticks(features[:15])
plt.show()

def pca_transform(n_comp):
    pca = PCA(n_components=n_comp, random_state=123)
    global Data_reduced
    Data_reduced = pca.fit_transform(Data)
    print('Shape of the new Data df: ' + str(Data_reduced.shape))

pca_transform(n_comp=1)
start8=time.time()
start9 = time.process_time()
k_means(n_clust=2, data_frame=Data_reduced, true_labels=Labels_binary)
end8 = time.time()
end9 = time.process_time()
clock0 = end8 - start8
cpu0 = end9-start9
print("wall-clock time is" + str(clock0) + "s")
print("run-time is" + str(cpu0) + "s")


pca_transform(n_comp=2)
k_means(n_clust=2, data_frame=Data_reduced, true_labels=Labels_binary)

