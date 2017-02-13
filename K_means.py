import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from numpy import *
import pickle
from sklearn import tree
import os.path
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.cm as cm
import time
from sklearn.cross_validation import KFold

features=['X Coordinate','Y Coordinate','Primary Type']
x_centroids=[]
y_centroids=[]

Number_of_Clusters=10

def clustering(newdf,x_centroids,y_centroids):
	datapointslist=[]
	for xpoint,ypoint in zip(newdf['X Coordinate'],newdf['Y Coordinate']):
		counter=0.0
		value=float('+inf')
		for xcentroid,ycentroid in zip(x_centroids,y_centroids):
			counter=counter+1
			distance=math.sqrt(math.pow(xcentroid-xpoint,2)+math.pow(ycentroid-ypoint,2))
			if distance<=value:
				value=distance
				cluster=counter
				#datapoints_with_cluster = [{'Cluster_Number':cluster,'X_Point':xpoint,'Y_Point':ypoint }]
		datapointslist.append([cluster,xpoint,ypoint])
	datapoints_dataframe=pd.DataFrame(datapointslist,columns=['Cluster_Number','XPoint','YPoint'])
	return datapoints_dataframe

	
def new_centroids(Datapoints_with_clusters,x_centroids,y_centroids,Number_of_Clusters):
	New_Centroid_Points = Datapoints_with_clusters.groupby('Cluster_Number').mean()
	counter_number=0.0
	print New_Centroid_Points
	for row in New_Centroid_Points.itertuples():
		for i in range(0,Number_of_Clusters):
			if x_centroids[x_centroids == x_centroids[i]].index[0]==row.Index.astype(np.int64)-1:
				if x_centroids[i]==row.XPoint and y_centroids[i]==row.YPoint:
					x_centroids[i]=row.XPoint
					y_centroids[i]=row.YPoint
				else:
					counter_number=counter_number+1
					x_centroids[i]=row.XPoint
					y_centroids[i]=row.YPoint
	return counter_number,x_centroids,y_centroids
	


def K_means():
	
	fig = plt.figure()
	df = pd.read_csv("Crimes_-_2001_to_present.csv",index_col=None,header=0,usecols=features)
	newdf=df.dropna()
	number_of_iterations=0.0
	
	x=newdf['X Coordinate'].unique()
	x=np.sort(x)
	y=newdf['Y Coordinate'].unique()
	y=np.sort(y)
	
	initial_random_centroids = newdf.sample(n=Number_of_Clusters)
	initial_random_centroids=initial_random_centroids.reset_index()
	initial_random_centroids=initial_random_centroids[['X Coordinate','Y Coordinate']]
	x_centroids=initial_random_centroids['X Coordinate']
	y_centroids=initial_random_centroids['Y Coordinate']
	print initial_random_centroids
	Datapoints_with_clusters=clustering(newdf,x_centroids,y_centroids)
	print Datapoints_with_clusters
	#new_centroids(Datapoints_with_clusters,x_centroids,y_centroids,Number_of_Clusters)
	counter_number,x_centroids,y_centroids=new_centroids(Datapoints_with_clusters,x_centroids,y_centroids,Number_of_Clusters)
	print x_centroids,y_centroids
	while counter_number!=0:
		number_of_iterations=number_of_iterations+1
		Datapoints_with_clusters=clustering(newdf,x_centroids,y_centroids)
		print Datapoints_with_clusters
		counter_number,x_centroids,y_centroids=new_centroids(Datapoints_with_clusters,x_centroids,y_centroids,Number_of_Clusters)
		if counter_number==0:
			break
	print Datapoints_with_clusters
	print x_centroids
	print y_centroids
	print number_of_iterations
	
	plt.xlim(x[1],x[-1])
	plt.ylim(y[1],y[-1])
	#plt.scatter(newdf['X Coordinate'],newdf['Y Coordinate'])
	
	colors =cm.rainbow(np.linspace(0, 1,Number_of_Clusters)) 
	
	for n,color in zip(range(1,Number_of_Clusters+1),colors):
		Datapoints_within_a_cluster=Datapoints_with_clusters[(Datapoints_with_clusters['Cluster_Number']==n)]
		plt.scatter(Datapoints_within_a_cluster['XPoint'],Datapoints_within_a_cluster['YPoint'],color=color)  #plotting datapoints
	plt.scatter(x_centroids,y_centroids,color='black')  #plotting centroids
	plt.show()	
	plt.close()
	
K_means()

