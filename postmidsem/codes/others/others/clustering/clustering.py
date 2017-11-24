import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
### For the purposes of this example, we store feature data from our
### dataframe `df`, in the `f1` and `f2` arrays. We combine this into
### a feature matrix `X` before entering it into the algorithm.
plt.rcParams['figure.figsize']=(16,9)
plt.style.use('ggplot')
dataset = np.loadtxt("xclara.csv", delimiter=",")
#print(dataset.shape)
X=dataset[:,0]
Y=dataset[:,1]
#print(X)
#print(Y)
#df=pd.read_csv("textfile.csv")
#print(df.shape)
#df.head()
#f1 = df['V1'].values
#f2 = df['V2'].values
#print(f1)
Z=np.array(list(zip(X,Y)))
lis=[]
#print(kmeans.inertia_)
#print(Z)
for i in range(1,11):
	inert=0
	kmeans = KMeans(n_clusters=i).fit(Z)
	r=kmeans.inertia_
	print(r)
	lis.append(r)
plt.plot([1,2,3,4,5,6,7,8,9,10],lis)	
plt.show()
#print(kmeans)
#c_center=kmeans.cluster_centers_
#print(c_center)
#label=kmeans.labels_[0]
#label2=kmeans.labels_[1]
#cx=c_center[:,0]
#cy=c_center[:,1]
	
#plt.scatter(X,Y,s=7,c='black')
#plt.scatter(X,Y,marker='.',s=100,c=kmeans.labels_)
#plt.scatter(cx,cy,marker="*",s=200,c='black')