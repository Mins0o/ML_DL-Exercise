from os import listdir
from numpy import average
import csv
import random
	
def TsvToData(filePath):
	with open(filePath,'r') as file:
		lines = list(csv.reader(file, delimiter = '\t'))
		data = [[int(x) for x in line[0].split(',')] for line in lines]
		label = [line[1] for line in lines]
	return(data, label)
	
def LoadFiles(filePath = None):
	if filePath == None:
		path = input("What is the path of your data folder?\n>>> ")
	else:
		path = filePath
	dataFiles = [file for file in listdir(path) if file[-4:]==".tsv"]
	for fileNum in range(len(dataFiles)):
		print("{0:02d}\t{1}".format(fileNum,dataFiles[fileNum]))
	selections = [int(x) for x in input("Type in indices of files, each separated by spacing\n>>> ").split()]
	filesDict = {}
	for selection in selections:
		filesDict[dataFiles[selection]] = TsvToData(path+"\\"+dataFiles[selection])
	return(filesDict)

def TruncateToMinLength(dataCollection):
	"""This method matches the length of the data by cutting off the tails of longer files"""
	# Get minimum length and file name of it
	minLength = 9999999
	fileName = ""
	for name in dataCollection:
		data = dataCollection[name][0]
		for singleDataStream in range(len(data)):
			if len(data[singleDataStream])<minLength:
				minLength = len(data[singleDataStream])
				fileName = "{0}, Line {1}".format(name, singleDataStream)
	
	# Confirm user action
	userAnswer = ""
	while not(userAnswer.lower() == "y" or userAnswer.lower() == "n"):
		userAnswer = input("The minimum length is {0} from {1}. Would you like to truncate the data?(Y/N)\n>>> ".format(minLength, fileName))
	
	# Slice and return
	if userAnswer.lower() == "y":
		output = ([], [])
		for dataFile in dataCollection:
			for i in range(len(dataCollection[dataFile][0])):
				output[0].append(dataCollection[dataFile][0][i][:minLength])
				output[1].append(dataCollection[dataFile][1][i])
	return output

def SaveData(data, filePath = None):
	if filePath == None:
		path = input("What is the path of your data folder?\n>>> ")
	else:
		path = filePath
	with open(path + "\\Truncated.tsv",'w') as file:
		for lineNumber in range(len(data[0])):
			file.write(",".join([str(x) for x in data[0][lineNumber]]) + "\t" + data[1][lineNumber] + "\n")
	print("Saved the truncated and combined file")

def MatchFrequency(data, originalF, targetF):
	if originalF > targetF:
		return DecreaseFrequency(data, originalF, targetF)
	elif originalF < targetF:
		return IncreaseFrequency(data, originalF, targetF)
	else:
		return data

def IncreaseFrequency(data, originalF, targetF):
	"""This method uses interpolation to fill in the gaps"""
	baseStep = targetF // originalF
	randomAddPossibility = targetF % originalF
	returnData = []
	index = 0
	endOfList = False
	randAdd = [1 for i in range(randomAddPossibility)] + [0 for i in range(originalF - randomAddPossibility)]
	while not endOfList:
		random.shuffle(randAdd)
		for randomArrayIndex in range(originalF):
			try:
				returnData += interpolate(data[index], data[index + 1], baseStep + randAdd[randomArrayIndex])
			except IndexError:
				endOfList = True
				break
			index += 1
	return(returnData)
			

def interpolate(point1, point2, numberOfPoints):
	"""<numberOfPoints> should be greater or equal to 1.
	<numberOfPoints> is number of points from point1 until point2."""
	if numberOfPoints == 1:
		return([point1])
	interval = (point2 - point1) / numberOfPoints
	return([point1 + i * interval for i in range(numberOfPoints)])
	

def DecreaseFrequency(data, originalF, targetF, avgOption = True):
	"""Decrease frequency by sampling from original data.
	This method uses psuedo random distribution to ensure it has rather uniform smapling rate match.
	With avgOption on(True), the sampling will use the average of the missed datapoints.
	If the option is False, it will sample from a single point."""
	baseStep = originalF // targetF
	randomAddPossibility = originalF % targetF
	returnData = []
	index = 0
	endOfList = False
	randAdd = list([1 for i in range(randomAddPossibility)] + [0 for i in range(targetF-randomAddPossibility)])
	if avgOption:
		prev = 0
		while not endOfList:
			random.shuffle(randAdd)
			for randomArrayIndex in range(targetF):
				slice = data[prev:index]
				if not slice == []:
					returnData.append(average(slice))
				else:
					endOfList = True
					break
				prev = index
				index += baseStep + randAdd[randomArrayIndex]
	else:
		while not endOfList:
			random.shuffle(randAdd)
			for randomArrayIndex in range(targetF):
				try:
					returnData.append(data[index])
				except IndexError:
					endOfList = True
					break
				index += baseStep + randAdd[randomArrayIndex]
	return returnData
	
if (__name__ == "__main__"):
	filePath = input("What is the path of your data folder?\n>>> ")
	SaveData(TruncateToMinLength(LoadFiles(filePath)), filePath)
	