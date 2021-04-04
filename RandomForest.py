#!/usr/bin/env python
# coding: utf-8

# In[26]:


# Random Forest Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('MyDataSet.csv')
x = dataset.iloc[:, 1].values.reshape((-1,1))
y = dataset.iloc[:, 0].values
# Training the Random Forest Regression model on the whole dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#using random forest regression algorithm
trainer = RandomForestRegressor(n_estimators = 5, random_state = 0)
trainer.fit(x_train, y_train)

# Predicting a new result
prediction = trainer.predict(x_test)
accuracy =  metrics.mean_absolute_error(y_test,prediction)
print(accuracy)

# Visualising the Random Forest Regression results (higher resolution)
X_axis = np.arange(min(x), max(x), 0.1)
X_axis = X_axis.reshape((len(X_axis), 1))
plt.scatter(x_train, y_train, color = 'red')
plt.plot(X_axis, trainer.predict(X_axis), color = 'blue')
plt.title(' Humidity-Temprature by(Random Forest Regression)')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.show()

###############################################################
X_axis = np.arange(min(x), max(x), 0.1)
X_axis = X_axis.reshape((len(X_axis), 1))
plt.scatter(x_test, y_test, color = 'green')
plt.plot(X_axis, trainer.predict(X_axis), color = 'k')
plt.title(' Humidity-Temprature by(Random Forest Regression)')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.show()

#exporting prediction to csv

humidity =  trainer.predict(x_test)
temperature = x_test[:,0]
Humidity_predictions = {'Humidity Predictions': humidity,
                       'Temperature': temperature}
predictions = pd.DataFrame(Humidity_predictions, columns =['Humidity Predictions','Temperature'])
print(predictions)
predictions.to_csv('Humidity Predictions(Random forest).csv',index =False)


# In[ ]:




