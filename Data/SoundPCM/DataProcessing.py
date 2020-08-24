from os import listdir
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
	pass

def IncreaseFrequency(data, originalF, targetF):
	pass

def DecreaseFrequency(data, originalF, targetF):
	baseStep = originalF // targetF
	randomAddPossibility = originalF % targetF
	returnData = []
	endOfList = False
	index = 0
	while not endOfList:
		try:
			returnData.append(data[index])
		except IndexError:
			endOfList = True			
		if random.randrange(0, targetF) >= randomAddPossibility:
			randomAddition = 0
		else:
			randomAddition = 1
		index = index + baseStep + randomAddition
	return returnData
	
if (__name__ == "__main__"):
	filePath = input("What is the path of your data folder?\n>>> ")
	SaveData(TruncateToMinLength(LoadFiles(filePath)), filePath)
	