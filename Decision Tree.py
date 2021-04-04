#!/usr/bin/env python
# coding: utf-8

# In[7]:


#training using decision tree
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import metrics
# Importing the dataset
dataset = pd.read_csv('MyDataSet.csv')

#independent var 1 temperature
x = dataset.iloc[:, 1:4].values
#dependent variable relative humidity
y = dataset.iloc[:, 0].values

#defining test and train data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# Training the Decision Tree Regression model on the whole dataset
tree = DecisionTreeRegressor()
tree.fit(x_train,y_train)

# Predicting a new result
prediction = tree.predict(x_test)
accuracy = metrics.mean_absolute_error(y_test,prediction)
print(accuracy )
# Visualising the Decision Tree Regression prediction results 
plt.plot(dataset,color='blue') 
plt.scatter(y_test,prediction,color='red')
plt.title('humidity (Decision Tree Regression)')
plt.xlabel('Temperature')
plt.ylabel('Relative Humidity')
plt.show()

#visualising wind speed and wind direction

plt.plot(x_test[:,-1],color = 'yellow')
plt.plot(y_test,color = 'brown')
plt.title('Wind direction vs Humidity')
plt.show()

plt.plot(dataset.iloc[:,2])
plt.scatter(x_test[:,2],y_test,color='orange')
plt.title('Wind speed vs Humidity')
plt.show()
#rounding temperature , wind speed & wind direction to 2 number of 10th grade

temperature = x_test[:,1]
wind_speed = x_test[:,2]
wind_direction = x_test[:,-1]
#rounding temperature
temp = np.array(temperature)
Temp_to_100 = np.around(temp,2)
Temp = list(Temp_to_100)
#rounding wind speed
Wind_Speed = np.array(wind_speed)
speed_to_100 = np.around(Wind_Speed,2)
WindSpeed = list(speed_to_100)
#rounding wind Direction
Wind_Direction = np.array(wind_direction)
direction_to_100  = np.around(Wind_Direction)
WindDirection = list(direction_to_100)
#expoting prediction to a csv file
humidity = prediction
predictions = {'Humidity Prediction': humidity,
              'Temperature': Temp,
              'Wind Speed': WindSpeed,
              'Wind Direction':WindDirection }
file = pd.DataFrame(predictions, columns =['Humidity Prediction','Temperature','Wind Speed','Wind Direction'])
file.to_csv('Humidity_Prediction.csv',index = False)


# In[ ]:




