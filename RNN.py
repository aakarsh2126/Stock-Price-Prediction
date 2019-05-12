#Recurrent Neural Network

#Part 1: Data Preprocessing
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the training set
dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
#independent variable vector 
training_set=dataset_train.iloc[:,1:2].values

#Feature Scaling
#Applying Normalisation
from sklearn.preprocessing import MinMaxScaler
#Creating Object of this class(all values in training set will be 0 and 1)
sc=MinMaxScaler(feature_range=(0,1))
#Creating scales training set
training_set_scaled=sc.fit_transform(training_set)

#Creating data structures with 60 timesteps and 1 output
#means at each step the RNN will look at 60 time units back then next output is predicted
#intializing X_train and y_train which will contain 60 previous stock prices
X_train=[]
y_train=[]
#Now appending scaled trained data from i-60 to i in X_train and i+1 output for y_train
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
#Now converting them in numpy array
X_train=np.array(X_train)
y_train=np.array(y_train)

#Reshaping(Adding new dimension.now we have a 3D tensor)
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


#Part 2: Building the RNN
#importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM

#intializing the RNN
#Regressor as we are predicting continous values
regressor=Sequential()
#Adding First LSTM layer with some Dropout regularization
#adding number of neurons
#adding boolean return sequences(true if we will add further LSTM layer otherwise false)
#adding input shape that contains last two arguments  of 3D tensor(as it is 1st layer)
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
#Dropout Regularization(20% neurons will be ignored)
regressor.add(Dropout(0.2))

#Adding  Second LSTM layer with some Dropout regularization
#no need to add input_shape() this time
regressor.add(LSTM(units=50,return_sequences=True))
#Dropout Regularization(20% neurons will be ignored)
regressor.add(Dropout(0.2))

#Adding Third LSTM layer with some Dropout regularization
regressor.add(LSTM(units=50,return_sequences=True))
#Dropout Regularization(20% neurons will be ignored)
regressor.add(Dropout(0.2))

#Adding Fourth LSTM layer with some Dropout regularization
#As last LSTM layer return_sequences=false(default)
regressor.add(LSTM(units=50))
#Dropout Regularization(20% neurons will be ignored)
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units=1))

#Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

#using tensorboard for storing logs
from keras.callbacks import TensorBoard
losses=TensorBoard(log_dir='./logs',batch_size=32)

#Fitting the RNN to the training set
regressor.fit(X_train,y_train,epochs=100,batch_size=32,callbacks=[losses])

#save model
regressor.save('model.h5')