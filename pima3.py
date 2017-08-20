#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 02:55:56 2017

@author: adi
"""

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
# seed for reproducing same results
seed = 9
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt('dataset.csv', delimiter=',')
#randomize the data set
np.random.shuffle(dataset)
# split into input and output variables
#X is the input data set
#Y is the target data set
X = dataset[:,0:8]
Y = dataset[:,8]
#scalar is used for normalization of input data set
scaler = StandardScaler()
X = scaler.fit_transform(X)
# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)

# create the model
#model used is sequential model because it grows in forward direction 
#We add first hidden layer having input dimension 8 in this case and then output dimension
model = Sequential()
model.add(Dense(800, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile the model
#parameters of the compile model are all default and optimize result for pima
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
#model.fit(X, Y, nb_epoch=100, batch_size=10,  verbose=2) # 150 epoch, 10 batch size, verbose = 2
#we use raining set and test set in this case to carry out computation of validation accuracy for each epoch taking batch_size as 10
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=150, batch_size=10, verbose=2)

# evaluate the model
#scores = model.evaluate(X, Y)
#evaluation is done on the test set
scores = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))