#!/usr/bin/env python
# coding: utf-8

# In[19]:


#using the 3rd algorithm to predict the weather
#k-neerest neighbor knn
############################################################
#1st importing you liberraires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics

#reading dataset
mydataset = pd.read_csv('MyDataSet.csv')

#defining the independent & dependent variables
#Temperature
x = mydataset.iloc[:,1].values.reshape((-1,1))

#Humidity
y = mydataset.iloc[:,0].values

#splitting data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#applying the algorithm

trainer = KNeighborsClassifier(n_neighbors=5)
trainer.fit(x_train, y_train)

#prediction
prediction = trainer.predict(x_test)
accuracy = metrics.mean_absolute_error(y_test,prediction)
print(round(accuracy,2) )
#creating confusion matrix
cm = confusion_matrix(y_test, prediction)

#plotting result of trining and testing  
X_axis = np.arange(min(x_train),max(x_train),0.1)
X_axis = X_axis.reshape((len(X_axis), 1))


plt.scatter(x_train,y_train,color='red')
plt.plot(X_axis,trainer.predict(X_axis),color='blue')
plt.title('K-NN (Training set)')
plt.xlabel('Temperature')
plt.ylabel('Relative Humidity')
plt.legend()
plt.show()

#test reult
X_axis = np.arange(min(x_test),max(x_test),0.1)
X_axis = X_axis.reshape((len(X_axis), 1))
##############################################
plt.xlim(x_test.min(), x_test.max())
plt.ylim(y_test.min(), y_test.max())
plt.scatter(x_test,y_test,color='green')
plt.plot(X_axis,trainer.predict(X_axis),color='k')
plt.title('K-NN (Training set)')
plt.xlabel('Temperature')
plt.ylabel('Relative Humidity')
plt.legend()
plt.show()
#export to csv file

humidity = trainer.predict(x_test)
temperatures = x_test[:,0]
humidity = {'Humidity Predictions': humidity,
           'Temperature': temperatures}
print(humidity)
predictions = pd.DataFrame(humidity,columns =['Humidity Predictions','Temperature'])
predictions.to_csv('Humidity(knn).csv',index=False)


# In[ ]:




