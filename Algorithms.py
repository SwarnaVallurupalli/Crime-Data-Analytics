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



df1=pd.DataFrame()
changedtimelist=[]


def data_preprocessing():
	
	if os.path.isfile("dataset.p"):
		df=pd.read_pickle("dataset.p")
	else:
	
		df = pd.read_csv("Crimes_-_2001_to_present.csv",index_col=None,header=0)
		df=df.dropna()
		df['Date_Format'],df['Time'],df['AM/PM']=df['Date'].str.split(' ').str
		df['Time_Hour']=df['Time'].str.split(':').str[0]
		df['Time_Minutes']=df['Time'].str.split(':').str[1]
		df['Time']=df['Time_Hour']+':'+df['Time_Minutes']
		df['Time']=df['Time']+' '+df['AM/PM']
		df['MONTH'],df['DAY'],df['YEAR']=df['Date_Format'].str.split('/').str
		#df[(df['Primary Type'][df.Primary Type=='NON - CRIMINAL']='NON-CRIMINAL'
		#df.Primary Type[df.Primary Type=='NON-CRIMINAL (SUBJECT SPECIFIED)']='NON-CRIMINAL'
		df['Primary Type'] = df['Primary Type'].astype('category')
		df['Primary Type']=df['Primary Type'].cat.codes
		df=df.reset_index()
		timeformat_conversion=df['Time']
		for i in range(0,len(df)):
			changedtime=time.strftime('%H:%M', time.strptime(timeformat_conversion[i], '%I:%M %p'))
			changedtimelist.append(changedtime)
		df['Time']=[t for t in changedtimelist]
		df['Time_Hour']=df['Time'].str.split(':').str[0]
		df['Time_Minutes']=df['Time'].str.split(':').str[1]
		df=df.convert_objects(convert_numeric=True)
		df['Time_in_Mintues']=[d*60 for d in df['Time_Hour']]+df['Time_Minutes']
		df.to_pickle("dataset.p")
	return df
	
df=data_preprocessing()
	
finaltraindataset=df[(df['Year']>=2001)&(df['Year']<=2015)]
TrainData=finaltraindataset[['X Coordinate','Y Coordinate','Year','MONTH','DAY','Time_in_Mintues']]
	
std_scale = preprocessing.StandardScaler().fit(TrainData)
df_std = std_scale.transform(TrainData)
TrainData=pd.DataFrame(df_std)
	
traindata_target=finaltraindataset['Primary Type']
TrainData_Target = traindata_target.as_matrix()
	
TestDataset=df[(df['Year']==2013)]
TestData=TestDataset[['X Coordinate','Y Coordinate','Year','MONTH','DAY','Time_in_Mintues']]
	
	
df_std1 = std_scale.transform(TestData)
TestData=pd.DataFrame(df_std1)
	
Actual_values=TestDataset['Primary Type']
#Actual_values=Actual_values.as_matrix
Number_of_samples=len(df)


def logistic_regression():
	logistic_reg = linear_model.LogisticRegression(C=1e5) #creting a model object for logistic regression
	logistic_reg.fit(TrainData,TrainData_Target)
	logistic_predictions=logistic_reg.predict(TestData) 	#Predicting the class label of Test Data
	Accurate_score=accuracy_score(logistic_predictions,Actual_values)
	print logistic_predictions
	print Accurate_score
	print logistic_reg
	return
	
logistic_regression()

def NeuralNetwork_Classifier():
	
	Neuralnetwork_Classifier = MLPClassifier(solver='sgd',activation='logistic',alpha=0.000001,learning_rate_init=0.001,learning_rate='adaptive', hidden_layer_sizes=45)
	Neuralnetwork_Classifier.fit(TrainData,TrainData_Target)
	Neural_predictions=Neuralnetwork_Classifier.predict(TestData)
	Accurate_score=accuracy_score(Neural_predictions,Actual_values)
	print Accurate_score
	print Neural_predictions
	print Neuralnetwork_Classifier
	return
	
NeuralNetwork_Classifier()
	
def DecisionTreeClassifier():
	DecisionTree_Classifier=tree.DecisionTreeClassifier()
	DecisionTree_Classifier.fit(TrainData,TrainData_Target)
	Decisiontree_predictions=DecisionTree_Classifier.predict(TestData)
	
	Accurate_score=accuracy_score(Decisiontree_predictions,Actual_values)
	print Accurate_score
	print DecisionTree_Classifier
	print Decisiontree_predictions
	print Actual_values

DecisionTreeClassifier()


def crossvalidation():
	neural_model = MLPClassifier(solver='sgd',activation='logistic',alpha=0.00000001,learning_rate_init=0.2,learning_rate='adaptive')
	cv = KFold(Number_of_samples,n_folds=3)
	Total_TrainData=df[['X Coordinate','Y Coordinate','Year','MONTH','DAY','Time_in_Mintues']]
	Total_TrainData_Target=df['Primary Type']
	newtraindata=Total_TrainData.as_matrix()
	
	newtraindata_target=Total_TrainData_Target.as_matrix()
	accuracy_score = []
	accuracy=[]
	for train_cv, test_cv in cv:
		model = neural_model.fit(newtraindata[train_cv],newtraindata_target[train_cv])
		neuralnetwork_predictions=model.predict(newtraindata[test_cv])
		accuracy_score.append(model.score(newtraindata[test_cv],newtraindata_target[test_cv]))
		#accuracy.append(accuracy_score(neuralnetwork_predictions,newtraindata_target[test_cv]))
	mean_accuracy= np.array(accuracy_score).mean() 
	print accuracy_score
	#print accuracy
	return mean_accuracy

Mean_accuracy_crossvalidation=crossvalidation()	
print Mean_accuracy_crossvalidation	
	
	






