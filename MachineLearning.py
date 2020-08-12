import sklearn
import random
from nltk.corpus import names
from sklearn.naive_bayes import BernoulliNB

class MLExampleNames:
	
	def __init__(self):
		self.data=[(w,"M") for w in names.words("male.txt")]+[(w,"F") for w in names.words("female.txt")]
		self.clf=BernoulliNB()
		
	def TrainTestSplit(self, ratio=75):
		temp=self.data.copy()
		random.shuffle(temp)
		shuffledData=temp
		testIndexEnd=len(shuffledData)*ratio//100
		return(shuffledData[:len(shuffledData)*ratio//100],shuffledData[len(shuffledData)*ratio//100:])
		
	def XYSplit(self,_data):
		return([i[0] for i in _data],[i[1] for i in _data])
		
	def FeatureExtract(self, words):
		"""This function will return:
		For list of data - a list of features
		For a single datum - the features"""
		returningValue=None
		listOfFeatures=[self.F01]
		if(type(words)==list):
			returningValue=[[],[]]
			for word in words:
				featureDict={}
				featureList=[]
				for func in listOfFeatures:
					kind,feat=func(word)
					featureDict[kind]=feat
					featureList.append(feat)
				returningValue[0].append(featureDict)
				returningValue[1].append(featureList)
		elif(type(words)==str):
			for func in listOfFeatures:
				kind,feat=func(words)
				featureDict[kind]=feat
			returningValue=featureDict	
		return returningValue
			
	#Feature01
	def F01(self, word):
		featureName="Last Letter"
		featureResult=float(int.from_bytes(bytes(word[-1],"UTF8"),"big"))
		return(featureName,featureResult)
		
	#Feature02
	def F02(self, word):
		featureName="First two Letters"
		featureResult=float(int.from_bytes(bytes(word[0:2],"UTF8"),"big"))
		return(featureName,featureResult)
		
	def fit(self,trainData):
		self.clf=BernoulliNB()
		x=trainData[0]
		y=trainData[1]
		x_feat=self.FeatureExtract(x)[1]
		self.clf.fit(x_feat,y)
		
	def run(self):
		train,test=self.TrainTestSplit(90)
		self.fit(self.XYSplit(train))
		x_test,y_test=self.XYSplit(test)
		print(self.clf.score(self.FeatureExtract(x_test)[1],y_test))
		
if __name__=="__main__":
	MLEx=MLExampleNames()
	MLEx.run()
input()