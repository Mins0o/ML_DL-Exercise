import sklearn
import random
from nltk.corpus import names
from sklearn.naive_bayes import *
import matplotlib.pyplot as plt
import csv
import PCMFeature
import numpy as np
from os import listdir

class MLExample:

	def __init__(self):
		"""__init__ should include the things listed below:
		self.data
		self.clf
		self.clfList
		self.FeatureFuncList
		and
		def feature_n to go in the FeatureFuncList
		"""
		self.data = [(1, 0.3, "a"),(2, -0.1, "b")]
		self.clf1 = CategoricalNB()
		self.clfList = [self.clf1]
		self.FeatureFuncList = [(lambda x : x)]

	def TrainTestSplit(self, ratio = 75):
		"""returns the data, split in two.
		The data is still in shape of [(<data>, <label>)...(<data>, <label>)]"""
		temp = self.data.copy()
		random.shuffle(temp)
		shuffledData = temp
		testIndexEnd = len(shuffledData) * ratio // 100
		return(shuffledData[:len(shuffledData) * ratio // 100], shuffledData[len(shuffledData) * ratio // 100:])
				
	def XYSplit(self, _dataWLabel):
		"""This method unzips the data and splits <data> and <label> into two lists."""
		return([line[:-1][0] for line in _dataWLabel], [line[-1] for line in _dataWLabel])
	
	def FeatureExtract(self, x_data):
		"""This method performs a list of functions on the x_data given and returns a list of the results.
		If there were 3 functions, this method will give you a list of three elements."""
		dataInFeatures = []
		for datum in x_data:
			singleFeatureList = []
			for func in self.FeatureFuncList:
				feat = func(datum)
				singleFeatureList.append(feat)
			dataInFeatures.append(singleFeatureList)
		return(dataInFeatures)
	
	def Fit(self, clf, trainData):
		x, y = self.XYSplit(trainData)
		x_feat = self.FeatureExtract(x)
		clf.fit(x_feat, y)

	def Evaluate(self, clfChoice = 1, trainPercentage = 70, verbose = False):
		if(not type(clfChoice) == int):
			print("Evaluate() Error: The first argument(clfChoice) should be an integer")
			return
		try:
			clf = self.clfList[clfChoice]
		except IndexError:
			print("Evaluate() Error: clfChoice - Index out of range")
			clf = self.clfList[0]
		except:
			print("Something went wrong")
			return
		print("Evaluation")
		print("With {1}% of dataset, Training {0}".format(str(type(clf))[28:-2],trainPercentage))
		train, test = self.TrainTestSplit(trainPercentage)
		self.Fit(clf, train)
		labels = list(set(self.XYSplit(self.data)[1]))
		x_test, y_test = self.XYSplit(test)
		x_feat = self.FeatureExtract(x_test)
		try:
			predictions = clf.predict(x_feat)
		except IndexError:
			# https://github.com/scikit-learn/scikit-learn/pull/16326
			print("Index Error. Refer to github.com/scikit-learn/scikit-learn/pull/16326")
			return
			
		confusionMatrix = sklearn.metrics.confusion_matrix(y_test, predictions, labels)
		print("\t", end = "")
		for labelCaption in labels:
			print("{}\t".format(labelCaption), end = "")
		print()
		for trueLabel in range(len(confusionMatrix)):
			print("{}\t".format(labels[trueLabel]), end = "")
			for pred in confusionMatrix[trueLabel]:
				print("{}\t".format(pred), end = "")
			print()
		print("Accuracy: {}".format(sklearn.metrics.accuracy_score(y_test, predictions)))
		if verbose:
			try:
				display = sklearn.metrics.plot_confusion_matrix(clf, x_feat, y_test, labels = labels, cmap = plt.cm.Blues, normalize = 'true')
				plt.show()
			except IndexError:
				print("Displaying confusion matrics - Index Error")

	def PosNegAnalysis(self, targetLabels, clfChoice = 1, trainPercentage = 70):
		"""This method is almost a legacy method and probably won't be so usefull"""
		warningMsg = "\n"
		errorMsg = "\n"
		# Checking target label
		if targetLabels == []:
			warningMsg += "!!!!Warning!!!!\n`targetLabels` is empty"
			print(warningMsg)
		targetLabels = [l.lower() for l in targetLabels]
		
		# Checking classifier option
		if(not type(clfChoice) == int):
			print("Evaluate() Error: The first argument(clfChoice) should be an integer")
			return
		try:
			clf = self.clfList[clfChoice]
		except IndexError:
			print("Evaluate() Error: clfChoice - Index out of range")
			clf = self.clfList[0]
		except:
			print("Something went wrong")
			return
			
		# Setting up the data
		print("With {1}% of dataset, Training {0}".format(str(type(clf))[28:-2],trainPercentage))
		train, test = self.TrainTestSplit(trainPercentage)
		self.Fit(clf, train)
		x_test, y_test = self.XYSplit(test)
		x_feat = self.FeatureExtract(x_test)
			
		# Starting analysis
		tP = 0
		tN = 0
		fP = 0
		fN = 0
		
		"""I chose not to do this because of the index error issue mentioned below"""
		# predictions=clf.predict(x_feat)
		
		# For every instance of test data, count the pos-neg s
		for i in range(len(x_test)):
			prediction = ""
			try:
				prediction = clf.predict([x_feat[i]])[0].lower()
			except IndexError:
					# https://github.com/scikit-learn/scikit-learn/pull/16326
					print("Index Error. Refer to github.com/scikit-learn/scikit-learn/pull/16326")
			y_eval = y_test[i].lower()
			if (prediction in targetLabels and y_eval in targetLabels):
				tP += 1
			elif (prediction not in targetLabels and y_eval not in targetLabels):
				tN += 1
			elif (prediction in targetLabels and y_eval not in targetLabels):
				fP += 1
			elif (prediction not in targetLabels and y_eval in targetLabels):
				fN += 1
			else:
				print("Something went wrong")	
		# Among all actual positives, how much did I get correct?
		try:
			posRecall = tP / (tP + fN)
		except ZeroDivisionError:
			posRecall = 0
			errorMsg += "PosNegAnalysis() Error: tP = {0}, fN = {1}\n".format(tP, fN)
		try:
			negRecall = tN / (tN + fP)
		except ZeroDivisionError:
			negRecall = 0
			errorMsg += "PosNegAnalysis() Error: tN = {0}, fP = {1}\n".format(tN, fP)
		
		# Among all claimed positives, how much did I get correct?
		try:
			posPrecision = tP / (tP + fP)
		except ZeroDivisionError:
			posPrecision = 0
			errorMsg += "PosNegAnalysis() Error: tP = {0}, fP = {1}\n".format(tP, fP)
		try:
			negPrecision = tN / (tN + fN)
		except ZeroDivisionError:
			negPrecision = 0
			errorMsg += "PosNegAnalysis() Error: tN = {0}, fN = {1}\n".format(tN, fN)
		
		# Print out hard result
		print("\n\t\t__Predicted Label__\n True|\tTrue Positive: {0}\t\tFalse Negative: {3}\nLabel|\tFalse Positive: {2}\t\tTrue Negative: {1}".format(tP,tN,fP,fN))
		
		# Print out Recall and Precisions
		print("\nRecall is among all true labels and precision is among predicted labels")
		print("pRecall: {0:.3f}\tpPrecision: {2:.3f}\nnRecall: {1:.3f}\tnPrecision:{3:.3f}".format(posRecall,negRecall,posPrecision,negPrecision))
		
		# Print out counfusion Matrix
		print("\nConfusion Matrix based on recall:\nTL\\PL\tP\tN\nP\t{0:.2f}\t{1:.2f}\nF\t{2:.2f}\t{3:.2f}".format(posRecall,1-posRecall,1-negRecall,negRecall))
		
		# Print out accuracy
		print("\nAccuracy: {0:.3f}".format((tP+tN)/len(y_test)))
		print(warningMsg, end = "")
		print(errorMsg)

class MLExampleNames(MLExample):
	
	def __init__(self):
		"""initiates with the data and all available NB classifier in sklearn"""
		self.data=[(w.strip(),"M") for w in names.words("male.txt")]+[(w.strip(),"F") for w in names.words("female.txt")]
		self.clf1=BernoulliNB()
		self.clf2=CategoricalNB()
		self.clf3=ComplementNB()
		self.clf4=GaussianNB()
		self.clf5=MultinomialNB()
		self.clfList=[self.clf1,self.clf2,self.clf3,self.clf4,self.clf5]
		self.FeatureFuncList=[self.F01,self.F02]
			
	#Feature01
	def F01(self, word):
		"""Last Letter"""
		#featureResult=ord(word[-1])
		featureResult=ord(word[0][-1].lower())-32
		return featureResult
		
	#Feature02
	def F02(self, word):
		"""First two Letters"""
		#featureResult=float(int.from_bytes(bytes(word[0:2],"UTF8"),"big"))
		featureResult=int.from_bytes(bytes(word[0][:2].lower(),"UTF8"),"little")
		return featureResult
				
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

		
class MLExampleSound(MLExample):

	def __init__(self):
		"""1. Read the .tsv file of audio PCM
		2. Fill in the classifiers
		3. Put the classifiers in a list
		4. Declare a FeatureFuncList"""
		
		
		path = "./"
		dataFiles = [file for file in listdir(path+'Data/SoundPCM/') if file[-4:] == ".tsv"]
		for fileNum in range(len(dataFiles)):
			print("{0:02d}\t{1}".format(fileNum, dataFiles[fileNum]))
		#selection = 0
		selection = int(input("Type in index of the .tsv file\n>>> "))
		print("{0} selected\n______________________________".format(dataFiles[selection]))
		try:
			#rateInput = 45000
			rateInput = int(input("What is the sampling rate (Hz) of this data?\n>>> "))
		except:
			print("Sampling rate should be an integer in Hz")
			return
			
		# I know it's pronounced 'RED' and writen as "read".
		readedLines = PCMFeature.TsvToLine(path + 'Data/SoundPCM/' + dataFiles[selection])
		self.data = [([line, readedLines[line][0]], readedLines[line][-1])for line in range(len(readedLines))]
		"""Format of data:
		([index#, [PCM datapoint_0, ... , PCM datapoint_n]], label)
		"""
		
		self.clf1 = BernoulliNB()
		self.clf2 = CategoricalNB()
		self.clf3 = ComplementNB()
		self.clf4 = GaussianNB()
		self.clf5 = MultinomialNB()
		self.clfList = [self.clf1, self.clf2, self.clf3, self.clf4, self.clf5]
		self.FeatureFuncList = [self.F03, self.F02, self.F03]
		self.samplingRate = rateInput
		print("Reading and analyzing data. \nThis may take a while...")
		self.fourierData= [PCMFeature.FourierTransform(datum[0][1], self.samplingRate) for datum in self.data]
	
	def F01(self, singlePcmStream):
		"""Most dominant frequency
		Format of a singlePcmStream:
		[index#, [PCM datapoint_0, ... , PCM datapoint_n]]"""
		maxPoint = np.abs(self.fourierData[singlePcmStream[0]]).argmax()
		return(PCMFeature.FrequencyChart(self.samplingRate // 2)[maxPoint])
	
	def F02(self, singlePcmStream):
		"""Duration
		When does the wave become significant and insignificant?
		[index#, [PCM datapoint_0, ... , PCM datapoint_n]]"""
		PCM = PCMFeature.MaxNormalize(np.array(singlePcmStream[1]))
		start = len(PCM)
		finish = len(PCM)
		inThreshold = 0.2
		outThreshold = 0.2
		stdThreshold = 0.016
		localRadius = 40
		for index in range(len(PCM)):
			if index < start and abs(PCM[index]) > inThreshold:
				start = index
				index = min(index + len(PCM) // 10, len(PCM) - 1)
			if index > (start + localRadius) and index < finish and abs(PCM[index]) < outThreshold and PCM[index-localRadius:index].std() < stdThreshold:
				finish = index
		duration = finish - start
		return duration
	
	def F03(self, singlePcmStream):
		"""Most dominant frequency among mid range"""
		maxPoint = np.abs(self.fourierData[singlePcmStream[0]][50:150]).argmax()
		return(PCMFeature.FrequencyChart(self.samplingRate // 2)[maxPoint+10])
	
	def LabelDistribution(self):
		from nltk import FreqDist
		print(FreqDist(self.XYSplit(self.data)[1]).most_common(4))
		
if __name__ == "__main__":
	"""
	MLExN = MLExampleNames()
	for clfChoice in range(5):
		for trainingPercentage in range(10,91,10):
			MLExN.Run(clfChoice, trainingPercentage)
		print()
	# def Evaluate(self, clfChoice = 1, trainPercentage = 70, verbose = True)
	MLExN.PosNegAnalysis(["F"])
	MLExN.Evaluate(verbose = False)
	"""
	
	MLExS = MLExampleSound()
	# def Evaluate(self, clfChoice = 1, trainPercentage = 70, verbose = True)
	for i in range(4):
		MLExS.Evaluate(0, 85, verbose = False)
		MLExS.Evaluate(1, 85, verbose = False)
		MLExS.Evaluate(2, 85, verbose = False)
		MLExS.Evaluate(3, 85, verbose = False)
	input()
