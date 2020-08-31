import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

class DLExampleMnist:
	def __init__(self):
		batch_size = 128
		num_classes = 10
		epochs = 20

		mnist=tf.keras.datasets.mnist

		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = x_train.reshape(60000, 784)
		x_test = x_test.reshape(10000, 784)
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')
		x_train /= 255
		x_test /= 255
		print(x_train.shape[0], 'train samples')
		print(x_test.shape[0], 'test samples')

		# # convert class vectors to binary class matrices
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)
		print(y_train)


		model = keras.models.Sequential()
		model.add(Dense(512, activation='relu', input_shape=(784,)))
		model.add(Dropout(0.2))
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(num_classes, activation='softmax'))

		model.summary()

		model.compile(loss='categorical_crossentropy',
					  metrics=['accuracy'])

		history = model.fit(x_train, y_train,
							batch_size=batch_size,
							epochs=epochs,
							verbose=1,
							validation_data=(x_test, y_test))

		score = model.evaluate(x_test, y_test, verbose=0)
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