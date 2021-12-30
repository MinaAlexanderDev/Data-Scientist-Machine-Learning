# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
#y = dataset.iloc[:, 3].values

import scipy.cluster.hierarchy as sch
dendogram =sch.dendrogram(sch.linkage(X,method='ward'))


#using the dendogram to find the Optimal number of clusters
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5 , affinity='euclidean' ,linkage='ward')

y_hc =hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='blue',label='careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='red',label='standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='sensable')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='target')
#plt.scatter(hc.cluster_centers_[:,0],hc.cluster_centers_[:,1] , s=300 , c='black', label='centroids')
plt.title('cluster of customer')
plt.xlabel('Annual Income($)' )
plt.ylabel('Spending Score')
plt.legend()
plt.show()
"""
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""