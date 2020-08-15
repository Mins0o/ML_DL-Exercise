import sklearn
import random
from nltk.corpus import names
from sklearn.naive_bayes import *
import matplotlib.pyplot as plt
import csv

class MLExampleNames:
	
	def __init__(self):
		"""initiates with the data and all available NB classifier in sklearn"""
		self.data=[(w.strip(),"M") for w in names.words("male.txt")]+[(w.strip(),"F") for w in names.words("female.txt")]
		self.clf1=BernoulliNB()
		self.clf2=CategoricalNB()
		self.clf3=ComplementNB()
		self.clf4=GaussianNB()
		self.clf5=MultinomialNB()
		self.clfList=[self.clf1,self.clf2,self.clf3,self.clf4,self.clf5]
		
	def TrainTestSplit(self, ratio=75):
		"""returns the data, split in two.
		The data is still in shape of [(<name>,<lable>)...(<name>,<lable>)]"""
		temp=self.data.copy()
		random.shuffle(temp)
		shuffledData=temp
		testIndexEnd=len(shuffledData)*ratio//100
		return(shuffledData[:len(shuffledData)*ratio//100],shuffledData[len(shuffledData)*ratio//100:])
		
	def XYSplit(self,_data):
		"""This function unzips the data and splits <name> and <label> into two lists."""
		return([i[0] for i in _data],[i[1] for i in _data])
		
	def FeatureExtract(self, words):
		"""This function returns:
		For list of data (a list of strings) - a list of features
		For a single datum (a string) - the features for a single datum, ready to be used for predict()"""
		returningValue=None
		listOfFeatures=[self.F01,self.F02]
		if(type(words)==list):
			returningValue=[]
			for word in words:
				featureList=[]
				for func in listOfFeatures:
					feat=func(word)
					featureList.append(feat)
				returningValue.append(featureList)
		elif(type(words)==str):
			featureList=[]
			for func in listOfFeatures:
				feat=func(words)
				featureList.append(feat)
			returningValue=[featureList]
		return returningValue
			
	#Feature01
	def F01(self, word):
		"""Last Letter"""
		#featureResult=ord(word[-1])
		featureResult=ord(word[-1].lower())-32
		return featureResult
		
	#Feature02
	def F02(self, word):
		"""First two Letters"""
		#featureResult=float(int.from_bytes(bytes(word[0:2],"UTF8"),"big"))
		featureResult=int.from_bytes(bytes(word[:2].lower(),"UTF8"),"little")
		return featureResult
		
	def Fit(self,clf,trainData):
		x,y=self.XYSplit(trainData)
		x_feat=self.FeatureExtract(x)
		clf.fit(x_feat,y)
		
	def Run(self,clfChoice=1,trainPercentage=70):
		#print("Running accuracy test")
		clf=self.clfList[clfChoice]
		print("With {1}% of dataset, Training {0}".format(str(type(clf))[28:-2],trainPercentage),end="\t")
		train,test=self.TrainTestSplit(trainPercentage)
		self.Fit(clf,train)
		x_test,y_test=self.XYSplit(test)
		try:
			print(clf.score(self.FeatureExtract(x_test),y_test))
		except IndexError:
			# https://github.com/scikit-learn/scikit-learn/pull/16326
			print("Index Error. Refer to github.com/scikit-learn/scikit-learn/pull/16326")

	def Evaluate(self,clfChoice=1,trainPercentage=70,verbose=True):
		clf=self.clfList[clfChoice]
		print("Evaluation")
		print("With {1}% of dataset, Training {0}".format(str(type(clf))[28:-2],trainPercentage))
		train,test=self.TrainTestSplit(trainPercentage)
		self.Fit(clf,train)
		x_test,y_test=self.XYSplit(test)
		x_feat=self.FeatureExtract(x_test)
		if verbose:
			tM=0
			tF=0
			fM=0
			fF=0			
			for i in range(len(x_test)):
				# There are more female names
				prediction="F"
				try:
					prediction=clf.predict([x_feat[i]])[0]
				except IndexError:
						# https://github.com/scikit-learn/scikit-learn/pull/16326
						print("Index Error. Refer to github.com/scikit-learn/scikit-learn/pull/16326")
				if (prediction=="M" and y_test[i]=="M"):
					tM+=1
				elif (prediction=="F" and y_test[i]=="F"):
					tF+=1
				elif (prediction=="M" and y_test[i]=="F"):
					fM+=1
				elif (prediction=="F" and y_test[i]=="M"):
					fF+=1
				else:
					print("Something went wrong")	
			# Among all actual positives, how much did I get correct?
			maleRecall=tM/(tM+fF)
			femaleRecall=tF/(tF+fM)
			
			# Among all claimed positives, how much did I get correct?
			malePrecision=tM/(tM+fM)
			femalePrecision=tF/(tF+fF)
			
			# Print out hard result
			print("\n\t\t__Predicted Label__\n True|\tTrue Male: {0}\t\tFalse Female: {3}\nLabel|\tFalse Male: {2}\t\tTrue Female: {1}".format(tM,tF,fM,fF))
			
			# Print out Recall and Precisions
			print("\nRecall is among all true labels and precision is among predicted labels")
			print("mRecall: {0:.3f}\tmPrecision: {2:.3f}\nfRecall: {1:.3f}\tfPrecision:{3:.3f}".format(maleRecall,femaleRecall,malePrecision,femalePrecision))
			
			# Print out counfusion Matrix
			print("\nConfusion Matrix based on recall:\nTL\\PL\tM\tF\nM\t{0:.2f}\t{1:.2f}\nF\t{2:.2f}\t{3:.2f}".format(maleRecall,1-maleRecall,1-femaleRecall,femaleRecall))
			
			# Print out accuracy
			print("\nAccuracy: {0:.3f}".format((tM+tF)/len(y_test)))
		else:
			try:
				display=sklearn.metrics.plot_confusion_matrix(clf,x_feat,y_test,cmap=plt.cm.Blues,normalize='true')
				plt.show()
			except IndexError:
				print("Index Error")
		
			

class MLExampleSound:
	def __init__(self):
        """1. Read the .tsv file saved by the DoorOpener recording and return data format. Separate X (data) and Y (label)"""
            rawRead=list(csv.reader(open(data.tsv, 'r'), delimiter='\t'))
            self.dataSet=[[int(x) for x in line[0].split(",")] for line in rawRead],[line[1] for line in rawRead]
           

		
if __name__=="__main__":
	MLEx=MLExampleNames()
	for clfChoice in range(5):
		for trainingPercentage in range(10,91,10):
			MLEx.Run(clfChoice,trainingPercentage)
		print()
	MLEx.Evaluate(verbose=True)
	input()
