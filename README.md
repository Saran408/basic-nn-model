### EX NO:01
### DATE:
# <p align="center"> Developing a Neural Network Regression 

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![image](https://user-images.githubusercontent.com/75235427/187089091-2cce1477-ecd8-4d51-bfc6-540f7069ca06.png)


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

## PROGRAM

```python

#Importing Required Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Dataframe
df=pd.read_csv(r"C:\Users\Dell\OneDrive\Documents\Book1.csv")
x=df[['X']].values
y=df[['Y']].values

#Splitting Training and Testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)

#Preprocessing
scaler=MinMaxScaler()
scaler.fit(x_train)
scaler.fit(x_test)
x_train1=scaler.transform(x_train)
x_test1=scaler.transform(x_test)

#Model
ai_brain= Sequential([
    Dense(3,activation='relu'),
    Dense(2,activation='relu'),
    Dense(1)
])
ai_brain.compile(
     optimizer = 'rmsprop',
     loss='mse'
)

#Fitting Model
ai_brain.fit(x_train1,y_train,epochs=5000)

#Loss
loss_df=pd.DataFrame(ai_brain.history.history)

loss_df.plot()
ai_brain.evaluate(x_test1,y_test)

#New Data
x_n1=[[11]]
x_n1_1=scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```

## Dataset Information

![image](https://user-images.githubusercontent.com/75235427/187089334-c807fede-894d-4a97-8e0b-a33d7978f3d5.png)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75235427/187089435-2bb4027b-944c-4038-93ef-788095783913.png)



### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/75235427/187089372-4a8eb5b2-2bca-42ce-bcbc-133a7776f2ce.png)


### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75235427/187089392-fbd87f9e-f4fe-4d25-b455-746c4ba12bf7.png)


## RESULT
Thus the Neural network for Regression model is Implemented successfully.
