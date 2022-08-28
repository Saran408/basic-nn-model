### EX NO:01
### DATE:
# <p align="center"> Developing a Neural Network Regression 

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

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

Include screenshot of the dataset

## OUTPUT

### Training Loss Vs Iteration Plot

Include your plot here

### Test Data Root Mean Squared Error

Find the test data root mean squared error

### New Sample Data Prediction

Include your sample input and output here

## RESULT
