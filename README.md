# Stock-Price-Prediction
A Recurrent Neural Network model to predict current stock price of google.
## Libraries Used
Numpy
Matplotlib
Keras
Pandas
Pickle
## Recurrent Neural Network
### Introduction
The Recurrent Neural Network(RNN) is a special type of neural network which has memory(LSTM).In Classical feed forward neural network,the output of any any node depends on the current input.In RNN the output of current state not only depends on current input,but also the output from previous state.
### ht=f(xt*w+h(t-1)*wrec)
ht=output of current state
xt=input at current state
w=weight matrix for node in current state
h(t-1)=output of previous state
wrec=weight matrix for node in previous state
### Long Short Term Memory(LSTM)
The LSTM cells functions the working of neuron in RNN model.The long means they can store memory for long period of time and short means they have limited memory.
## Working 
The working of model is done by first training the model with dataset of google's 4 years stock price data(2012-2016).During training model uses output from previous 60 timestamps to predict output of current timestamp.we have one input layer,4 hidden layers each consisting 50 LSTM cells and one dense output layer.The dropout regularization is 20% to avoid overfitting.
once trained with the dataset, the model predicts the stock price for January 2017 and show the analysis between the predicted and actual results.
