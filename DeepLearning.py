import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import pickle

class DLExampleMnist:
	def __init__(self,batch_size = 10000, epochs = 20, verbose = False):
		# Batch size means how many data I will load and propagate.
		# For 60,000 data, 1 epoch will take 6 propagation for batch size of 10,000.
		# The network estimates with 10,000 datas and then adjust the weights
		
		# from 0 to 9
		num_classes = 10
		
		# load dataset
		mnist = tf.keras.datasets.mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		
		# Flattening the 2D image to a string of numbers.
		x_train = x_train.reshape(60000, 784)
		x_test = x_test.reshape(10000, 784)
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		
		# normalizing the pixel's values to 0 ~ 1. It was 8 bit brightness intensity before.
		x_train /= 255
		x_test /= 255
		
		if verbose:
			print(x_train.shape[0], 'train samples')
			print(x_test.shape[0], 'test samples')

		""" convert class vectors to binary class matrices
		One hot encoding.
		[0, 1, 2, 3, 4] => [[1, 0, 0, 0, 0],
		                    [0, 1, 0, 0, 0],
		                    [0, 0, 1, 0, 0],
		                    [0, 0, 0, 1, 0],
		                    [0, 0, 0, 0, 1]]
		This helps the cost function.
		https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
		"""
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)
		if verbose:
			print("y_train")
			print(y_train)


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
		model.add(Dense(512, activation='relu', input_shape = (784,)))
		model.add(Dropout(0.2))
		# layer 2
		model.add(Dense(200, activation='relu', input_shape = (512,)))
		model.add(Dropout(0.2))
		# last layer
		model.add(Dense(num_classes, activation='softmax'))

		model.summary()

		model.compile(loss='categorical_crossentropy',
					  metrics=['accuracy'])

		history = model.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=1,
							validation_data=(x_test, y_test))

		score = model.evaluate(x_test, y_test, verbose = False)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		
	def printFormattedArray(array):
		for i in range(len(array)):
			for j in range(len(array[i])):
				print(array[i][j],end=" | ")
			print()
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
	
if __name__ == "__main__":
	test = DLExampleMnist()