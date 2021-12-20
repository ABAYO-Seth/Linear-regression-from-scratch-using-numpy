import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
from sklearn.model_selection import train_test_split





class LinearRegressor():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        # Setting an initial value for the coefficients matrix
        self.coefficients = None
        self.predicted = None

    def train(self):
        "Computes the coefficients based on the dataset received by the model"

        # Train the model based on the data passed
        xt = np.transpose(self.X)
        prod = np.dot(xt,self.X)
        x_val = np.linalg.pinv(prod)
        y_val = np.dot(xt,self.y)
        self.coefficients = np.dot(x_val, y_val)

        return self.coefficients



    def predict(self, input):
        "Returns a prediction based on the learnt model and using the parameter passed"

        # Returns the prediction based on the learnt coefficient

        self.predicted = np.dot(input, self.coefficients)
        return self.predicted

        

    def getError(self):
        "Computes and returns the least square error for the dataset"

        # Computes the error rate for the dataset

        return np.power(LA.norm(self.y - self.predict(self.X)),2)


# # ==================DISCLAIMER=====================
# The dataset that will be used in this project was adapted from the Fish
# Market dataset on Kaggle (https://www.kaggle.com/aungpyaeap/fish-market).

# # The purpose of this study is to predict the weight of a fish using 5 explanatory variables (features) which are Length1, Length2, Length3, Height and Width

# In[20]:



#  Load the dataset from the Fish.csv file into the dataset variable below
dataset = np.genfromtxt('Fish.csv', delimiter=',')


#Retrieve the shape of the dataset
shape = dataset.shape
print("Shape of the dateset is:", shape)




#  Retrieve the features into the X array, and the output to the y array
  
X = dataset[1:,2:7]
y = dataset[1:,1:2]

# Apply min-max normalization on each of the columns of the X array 
for i in range(X.shape[1]):
      
      X[:,i]= (X[:,i] - X[:,i].min()) / (np.ptp(X[:,i]))

  # In real-life training a model requires a training and testing dataset.
  # In this stage, we will randomly generate the two datasets using 75% for the training dataset
  # and 25% for the testing dataset. We use the train_test_split function for this

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)

lr = LinearRegressor(X_train, y_train)
lr.train()

# Now we can test our model

# Make a prediction using the test dataset

predicted = lr.predict(X_test)


print("Least square root error = ", np.sqrt(lr.getError()))




# Visualization

# scatterplot of predicted weight against actual weight
# Plot a line dashed line y = x to visually appreciate how far the actual weight is from the predicted weight

# Plotting a scatter plot for the predicted and actual weights

plt.title("Graph for predicted and actual weights")
plt.scatter(y_test,predicted)
plt.xlabel("Actual weight")
plt.ylabel("Predicted Weight")

