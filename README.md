### EX NO: 01

### DATE: 29/08/2022


# <p align = "center"> Developing a Neural Network Regression Model </p>
 

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The Neural network model contains input layer,two hidden layers and output layer.Input layer contains a single neuron.Output layer also contains single neuron.First hidden layer contains six neurons and second hidden layer contains five neurons.A neuron in input layer is connected with every neurons in a first hidden layer.Similarly,each neurons in first hidden layer is connected with all neurons in second hidden layer.All neurons in second hidden layer is connected with output layered neuron.Relu activation function is used here .It is linear neural network model(single input neuron forms single output neuron).Explain the problem statement

##  </br></br></br></br></br></br></br></br></br></br></br></br></br></br></br></br>Neural Network Model
![Screenshot 2022-11-26 225802](https://user-images.githubusercontent.com/75235789/204101538-a9914978-db22-414d-b3bc-4619415e403f.jpg)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## </br>PROGRAM
```python
# Developed By : SARAN M
# Register Number : 212220230044

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_csv("data.csv")
df.head()

x=df[['input']].values
x

y=df[['output']].values
y

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=40)
scaler=MinMaxScaler()
scaler.fit(xtrain)
scaler.fit(xtest)
xtrain1=scaler.transform(xtrain)
xtest1=scaler.transform(xtest)

model=Sequential([
    Dense(6,activation='relu'),
    Dense(5,activation='relu'),
   
    Dense(1),
])


model.compile(optimizer='rmsprop',loss='mse')

model.fit(xtrain1,ytrain,epochs=5000)

lossmodel=pd.DataFrame(model.history.history)
lossmodel.plot()

model.evaluate(xtest1,ytest)

xn1=[[30]]
xn11=scaler.transform(xn1)
model.predict(xn11)
```

## Dataset Information
![Screenshot 2022-11-26 230353](https://user-images.githubusercontent.com/75235789/204101772-240b8325-4cb9-42ed-bdca-13e6c28c36dc.jpg)



## </br>OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2022-08-29 003447](https://user-images.githubusercontent.com/75235789/187090587-075e54c8-2ba4-4460-8e8a-555762e75fda.jpg)

### Test Data root Mean Squared Error
![Screenshot 2022-11-26 231534](https://user-images.githubusercontent.com/75235789/204102224-138e0acf-f161-4922-b845-30e6c104db37.jpg)


### New Sample Data Protection
![Screenshot 2022-08-29 003540](https://user-images.githubusercontent.com/75235789/187090615-08b2987e-0ca0-4b3d-86cd-500c956c2d33.jpg)

## RESULT

Thus, the neural network model regression model for the given dataset is developed.
