from keras.model import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(100,input_dim=128, init='uniform',activation='relu'))
model.add(Dense(100,init='uniform',activation='relu'))
model.add(Dense(1,init='uniform',activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train,validation_data=(X_test,Y_test),nb_epoch=500,batch_size=10,verbose=2)

#training accuracy
scores=model.evaluate(X_train,Y_train);
print("Accuracy: %.2f%%" % (scores[1]*100))

#testing accuracy
scores2=model.evaluate(X_test,Y_test)
print("Accuracy: %.2f%%" % (scores2[1]*100))