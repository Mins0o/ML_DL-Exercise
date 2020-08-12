import sklearn
import random
from nltk.corpus import names
from sklearn.naive_bayes import *

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
		
	def fit(self,clf,trainData):
		x,y=self.XYSplit(trainData)
		x_feat=self.FeatureExtract(x)
		clf.fit(x_feat,y)
		
	def run(self,clfChoice=1,trainPercentage=70):
		clf=self.clfList[clfChoice]
		print("With {1}% of dataset, Training {0}".format(str(type(clf))[28:-2],trainPercentage),end="\t")
		train,test=self.TrainTestSplit(trainPercentage)
		self.fit(clf,train)
		x_test,y_test=self.XYSplit(test)
		print(clf.score(self.FeatureExtract(x_test),y_test))

class MLExampleSound:
	def __init__(self):
		pass
		
if __name__=="__main__":
	MLEx=MLExampleNames()
	for clfChoice in range(5):
		print()
		print()
		for trainingPercentage in range(10,91,10):
			try:
				MLEx.run(clfChoice,trainingPercentage)
			except IndexError:
				# https://github.com/scikit-learn/scikit-learn/pull/16326
				print("Index Error. Refer to github.com/scikit-learn/scikit-learn/pull/16326")
	input()