#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Training using Simple Linear regression algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#importing dataset
MyDataSet = pd.read_csv('MyDataSet.csv')

#defining the independent & dependent variables
#independent vriables
x = MyDataSet.iloc[:,1].values.reshape((-1,1))
#dependent variable
y = MyDataSet.iloc[:,0].values

#splitting data 75/25
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#using our algorithm
trainer = LinearRegression()
trainer.fit(x_train,y_train)

# predicting
prediction = trainer.predict(x_test)
accuracy = metrics.mean_absolute_error(y_test,prediction)
print(round(accuracy,2) )
#plotting our training and testing results
plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,trainer.predict(x_train),color='red')
plt.xlabel("Temperature")
plt.ylabel("Humdity")
plt.show()


#showing results of test results
plt.scatter(x_test,y_test,color='green')
plt.plot(x_test,prediction,color='yellow')
plt.xlabel("temperature")
plt.ylabel("Humidity")
plt.show()

#extracing prediction as csv
humidity = prediction
temperature = x_test[:,0]
Humidity_predictions = {'Humidity prediction': humidity,
                       'Temperature':temperature }
print(Humidity_predictions)
predictions = pd.DataFrame(Humidity_predictions, columns = ['Humidity prediction','Temperature'])
predictions.to_csv('Humidity_predictions(Linear Regression).csv',index =False)


# In[ ]:




