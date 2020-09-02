import numpy as np
import pandas as pd
#import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import random
import matplotlib.pyplot as plt
import os
import PCMFeature
import sklearn
from sklearn.metrics import confusion_matrix

class DLExampleMnist:	
	def __init__(self):
		"""
		Constructor only loads the training data and testing data.
		Mnist is a classical dataset of handwritten numbers.
		Go to https://en.wikipedia.org/wiki/MNIST_database for more information.
		
		You can also take a look at an example with the method VisualizeData()
		"""
		
		# Load dataset
		mnist = tf.keras.datasets.mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		
		# Flattening the 2D image to a string of numbers.
		# normalizing the pixel's values to 0 ~ 1. It was 8 bit brightness intensity before.
		self.x_train = x_train.reshape(60000, 784).astype('float32') / 255
		self.x_test = x_test.reshape(10000, 784).astype('float32') / 255
	
		""" convert class vectors to binary class matrices
		One hot encoding.
		[0, 1, 2, 3, 4] => [[1, 0, 0, 0, 0],
		                    [0, 1, 0, 0, 0],
		                    [0, 0, 1, 0, 0],
		                    [0, 0, 0, 1, 0],
		                    [0, 0, 0, 0, 1]]
		This helps the cost function.(My speculation
		https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
		Here I use 10 because it is from 0 to 9."""
		self.y_train = keras.utils.to_categorical(y_train, num_classes = 10)
		self.y_test = keras.utils.to_categorical(y_test, num_classes = 10)
		
		# check data
		"""if verbose:
			print(self.x_train[0].shape(), 'train samples')
			print(self.x_test[0].shape(), 'test samples')
			print("y_train")
			print(self.y_train)	"""
		
	def LoadModel(self):
		path = input("What is the path to the directory the .mdel files are in?\n>>> ")
		modelFiles = [file[:-6] for file in os.listdir(path) if file[-6:] == ".index"]
		if len(modelFiles) == 0:
			print("No .mdel file found in this directory.")
			return
		for fileNum in range(len(modelFiles)):
			print("{0:02d}\t{1}".format(fileNum, modelFiles[fileNum]))
		selection = int(input("Type in index of the .mdel file\n>>> "))
		print("{0} selected\n______________________________".format(modelFiles[selection]))
		model = self.CreateModel()
		model.load_weights(path+"/"+modelFiles[selection])
		return model
	
	def VisualizeMnistData(self):
		for i in range(15):
			temp = plt.subplot(5, 3, i+1)
			pick = random.randrange(0,60000)
			temp.title.set_text(self.y_train[pick].argmax())
			pickedImg = (self.x_train[pick] * 255).astype('int32').reshape(28,-1)
			plt.axis('off')
			plt.imshow(pickedImg, cmap = 'Greys')
		plt.show()
	
	def CreateModel(self, dropRate = 0.2):
		# Batch size means how many data I will load and propagate.
		# For 60,000 data, 1 epoch will take 6 propagation for batch size of 10,000.
		# The network estimates with 10,000 datas and then adjust the weights
		model = keras.models.Sequential()
		"""
		                   layer1       layer2     last layer
		                     O            O            O
		                     O            O            O
		[][][]..[][][]  ==> ... 512   => ... 200   => ...   10  =>
		     -784-           O            O            O(num_classes)
		                     O            O            O
		"""
		# layer 1
		model.add(Dense(512, activation = 'relu', input_shape = (784,)))
		model.add(Dropout(dropRate))
		# layer 2
		model.add(Dense(200, activation = 'relu', input_shape = (512,)))
		model.add(Dropout(dropRate))
		# last layer
		model.add(Dense(10, activation = 'softmax'))

		model.summary()

		model.compile(loss = 'categorical_crossentropy',
					  metrics = ['accuracy'])
		return model
	
	def FitNewModel(self,
					batch_size = 5000,
					epochs = 20,
					dropRate = 0.2,
					verbose = 2):
		model = self.CreateModel(dropRate = dropRate)
		
		checkpointPath = "./models/saved.mdel"
		
		alternative = input("If you would like to save your model, type in the file name." +
							"ex)'test' or 'model_01'\nIf you do not wish to save the model," +
							" just leave the input as black and hit return.\n>>> ")
		if not(alternative == ""):
			checkpointPath = "./models/" + alternative + ".mdel"
			saveCheckpointCB = tf.keras.callbacks.ModelCheckpoint(filepath = checkpointPath,
                                                 save_weights_only = True,
                                                 verbose = 1)
			print("\n>>>>>>>>>>>>>Training in progress...<<<<<<<<<<<<<<<")
			history = model.fit(self.x_train, self.y_train,
							batch_size = batch_size,
							epochs = epochs,
							callbacks = [saveCheckpointCB],
							verbose = verbose)
		else:
			print("\n>>>>>>>>>>>>>Training in progress...<<<<<<<<<<<<<<<")
			history = model.fit(self.x_train, self.y_train,
							batch_size = batch_size,
							epochs = epochs,
							verbose = verbose)
		print(">>>>>>>>>>>>>>---Training finished---<<<<<<<<<<<<<<\n")
		return model
			
	def Evaluate(self, model):
		score = model.evaluate(self.x_test, self.y_test, verbose = 0)
		print('Test loss: ', score[0])
		print('Test accuracy: ', score[1])
	
	"""mnist=tf.keras.datasets.mnist

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.reshape(60000, 784)
	x_test = x_test.reshape(10000, 784)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	model = keras.models.Sequential()
	model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
	model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units = 10, activation = 'softmax'))"""

class DLExampleAudioPCM_woConv:
	def __init__(self):
		"""Constructor of this class only reads a data file.
		This class has only one property which is self.data"""
		#path = "./"
		path = "./"
		dataFiles = [file for file in os.listdir(path+'Data/SoundPCM/') if file[-4:] == ".tsv"]
		for fileNum in range(len(dataFiles)):
			print("{0:02d}\t{1}".format(fileNum, dataFiles[fileNum]))
		#selection = 0
		selection = int(input("Type in index of the .tsv file\n>>> "))
		print("{0} selected\n______________________________".format(dataFiles[selection]))
			
		readedLines = PCMFeature.TsvToLine(path + 'Data/SoundPCM/' + dataFiles[selection])
		self.data = [(readedLines[line][0], readedLines[line][-1])for line in range(len(readedLines))]
		"""Format of data:
		[([PCM datapoint_0, ... , PCM datapoint_n], label)...]
		"""

	def CreateModel(self, dropRate = 0.2):
		# Batch size means how many data I will load and propagate.
		# For 60,000 data, 1 epoch will take 6 propagation for batch size of 10,000.
		# The network estimates with 10,000 datas and then adjust the weights
		model = keras.models.Sequential()
		"""
		                   layer1       layer2     last layer
		                     O            O            O
		                     O            O            O
			[][][]..[][][]  ==> ... 500   => ... 200   => ...   4  =>
		     -inputShape-    O            O            O   (num_classes)
		                     O            O            O
		"""
		
		inputShape = len(self.data[0][0])
		# layer 1
		model.add(Dense(5000, activation = 'relu', input_shape = (inputShape,)))
		model.add(Dropout(dropRate))
		# layer 2
		model.add(Dense(2000, activation = 'relu', input_shape = (5000,)))
		model.add(Dropout(dropRate))
		# layer 3
		model.add(Dense(1500, activation = 'relu', input_shape = (2000,)))
		model.add(Dropout(dropRate))
		# layer 4
		model.add(Dense(1000, activation = 'relu', input_shape = (1000,)))
		model.add(Dropout(dropRate))
		# layer 5
		model.add(Dense(500, activation = 'relu', input_shape = (1000,)))
		model.add(Dropout(dropRate))
		# last layer
		model.add(Dense(4, activation = 'softmax'))

		model.summary()
		
		optimizer = keras.optimizers.Adam(lr = 0.0005)
		model.compile(loss = 'categorical_crossentropy',
					  metrics = ['accuracy'],
					  optimizer = optimizer)
		return model	
	
	def FitNewModel(self, epochs = 120, dropRate = 0.65,
					verbose = 2, trainTestSplit = 80):
		model = self.CreateModel(dropRate = dropRate)
		
		tempData = self.data.copy()
		random.shuffle(tempData)
		shuffledData = tempData
		
		#TrainTestSplit
		trainIndexEnd = len(shuffledData) * trainTestSplit //100
		trainLine = shuffledData[:trainIndexEnd]
		testLine = shuffledData[trainIndexEnd:]
		
		#XYSPlit
		x_train = np.array([PCMFeature.MaxNormalize(np.array(line[0])) for line in trainLine])
		y_train = [line[-1] for line in trainLine]
		x_test = np.array([PCMFeature.MaxNormalize(np.array(line[0])) for line in testLine])
		y_test = [line[-1] for line in testLine]
		
		batch_size = len(y_train)//2
		
		numericalLabel_train = []
		numericalLabel_test = []
		for i in y_train:
			if i == "t":
				numericalLabel_train.append(0)
			elif i == "s":
				numericalLabel_train.append(1)
			elif i == "c":
				numericalLabel_train.append(2)
			else:
				numericalLabel_train.append(3)
		for i in y_test:
			if i == "t":
				numericalLabel_test.append(0)
			elif i == "s":
				numericalLabel_test.append(1)
			elif i == "c":
				numericalLabel_test.append(2)
			else:
				numericalLabel_test.append(3)
		
		y_train = keras.utils.to_categorical(numericalLabel_train, num_classes = 4)
		y_test = keras.utils.to_categorical(numericalLabel_test, num_classes = 4)
		
		checkpointPath = "./models/saved.mdel"
		
		"""alternative = input("If you would like to save your model, type in the file name." +
							"ex)'test' or 'model_01'\nIf you do not wish to save the model," +
							" just leave the input as black and hit return.\n>>> ")"""
		if False and not(alternative == ""):
			checkpointPath = "./models/" + alternative + ".mdel"
			saveCheckpointCB = tf.keras.callbacks.ModelCheckpoint(filepath = checkpointPath,
                                                 save_weights_only = True,
                                                 verbose = 1)
			print("\n>>>>>>>>>>>>>Training in progress...<<<<<<<<<<<<<<<")
			history = model.fit(x_train, y_train,
							batch_size = batch_size,
							epochs = epochs,
							callbacks = [saveCheckpointCB],
							verbose = verbose,
							validation_data = (x_test,y_test))
		else:
			print("\n>>>>>>>>>>>>>Training in progress...<<<<<<<<<<<<<<<")
			history = model.fit(x_train, y_train,
							batch_size = batch_size,
							epochs = epochs,
							verbose = verbose,
							validation_data = (x_test,y_test))
		print(">>>>>>>>>>>>>>---Training finished---<<<<<<<<<<<<<<\n")
		return (model, (x_train, y_train), (x_test, y_test))
	
	def ConfusionMatrix(self, model, x, y):
		pred = [i.argmax() for i in model.predict(x)]
		trueL = [i.argmax() for i in y]
		confusionMatrix = confusion_matrix(trueL, pred, [0, 1, 2, 3])
		print("vT|P>\t", end = "")
		labels = ["t", "s", "c", "w"]
		for labelCaption in labels:
			print("{}\t".format(labelCaption), end = "")
		print()
		for trueLabel in range(len(confusionMatrix)):
			print("{}\t".format(labels[trueLabel]), end = "")
			for pred in confusionMatrix[trueLabel]:
				print("{}\t".format(pred), end = "")
			print()

	def Evaluate(self, model, test):
		score = model.evaluate(test[0], test[1], verbose = 0)
		print('Test loss: ', score[0])
		print('Test Accuracy: ', score[1])
		
if __name__ == "__main__":
	"""# Experiment with mnist
	mnistEx = DLExampleMnist()
	#mnistEx.VisualizeMnistData()
	model_MN_01 = mnistEx.FitNewModel(verbose = 2)
	mnistEx.Evaluate(model_MN_01)
	model_MN_02 = mnistEx.LoadModel()
	mnistEx.Evaluate(model_MN_02)"""
	
	# Experiment with PCM
	pcmEx = DLExampleAudioPCM_woConv()
	model_pcm_01, pcm_train, pcm_test = pcmEx.FitNewModel(trainTestSplit = 80)
	pcmEx.ConfusionMatrix(model_pcm_01, pcm_test[0], pcm_test[1])
	pcmEx.Evaluate(model_pcm_01, pcm_test)
	input("Press any key...")
	