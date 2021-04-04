#!/usr/bin/env python
# coding: utf-8

# In[1]:


#in this code I will try to make the prediction and training along side with all other stages 
#of preparating data in the same code & create a GUI to show the results
#1 importing liberaries
#wx is the gui library

#pandas is data manipulating lib 
#tkiner is for GUI
#PIL for image dispaly
#numpy for array manipulation and other numerical modification
#Think speak for importing data from channels
import thingspeak 
import requests
#regular expresion liberary used in finding match in strings
import re
import tkinter as tk
from tkinter import *
from PIL import ImageTk,Image 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#here we use more than one prediction technique by usng ensemble
from sklearn.ensemble import VotingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
#accurecy measure
from sklearn import metrics
from sklearn.metrics import r2_score
#importing plotting liberary
import matplotlib.pyplot as plt
#global variables for thing speak results
global read_key 
read_key = "IFY6WD7REA1PCCVC"
global channel_id
channel_id = 821324
global crops
global real_temperature
global real_humidity
global results
global field1
global field2
global channel
global soil_moisture
global real_soil_moisture

#importing dataset

MyDataSet = pd.read_csv('MyDataSet.csv')

#defining which is my dependent variables and independent variable

X = MyDataSet.iloc[:,1].values.reshape((-1,1))
Y = MyDataSet.iloc[:,0].values

#splittingdata with 25/75 percent rule
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

#using the ensembel technique
#decision Tree
tree = DecisionTreeRegressor()
tree.fit(x_train,y_train)
tree.predict(x_test)
#linear Regression
linearRegressorTrainer = LinearRegression()
linearRegressorTrainer.fit(x_train,y_train)
linearRegressorTrainer.predict(x_test)
#randome forest
RandomForestTrainer = RandomForestRegressor(n_estimators = 5,random_state=0)
RandomForestTrainer.fit(x_train,y_train)
RandomForestTrainer.predict(x_test)
#knn
KnnTrainer = KNeighborsRegressor(n_neighbors=15)
KnnTrainer.fit(x_train,y_train)
KnnTrainer.predict(x_test)
#now we call the ensembel classifier
Ensembelling = VotingRegressor(estimators =[('Decision Tree',tree),
                                            ('Knn',KnnTrainer),
                                             ('Linear Regression',linearRegressorTrainer),
                                            ('Random Forest',RandomForestTrainer)],
                                            )
Ensembelling.fit(x_train,y_train)
global prediction
prediction = Ensembelling.predict(x_test)
#error and accurecy
global error_percent
error_percent = metrics.mean_absolute_error(y_test,prediction)
global sucess_percent
sucess_percent = metrics.r2_score(y_test,prediction)
#extracting data an Xls sheet
humidity = prediction
temperature = x_test[:,0]
global Ensemble_Prediction
Ensemble_Prediction = {'Humidity_Predictions': humidity,
                      'Temperature': temperature}
Ensemble_prediction = pd.DataFrame(Ensemble_Prediction, columns = ['Humidity_Predictions','Temperature'])
Ensemble_prediction.to_csv('Ensemble_Predictions.csv', index=False)

#showing the plotting of test samples
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_test,prediction,color='red')
plt.xlabel("temperature")
plt.ylabel("Humidity")
plt.title("Ensembling")
plt.show()
####################################################
#thingspeak parameters
#######################################################
def get_field_last(self, field=None, options=None):
        """To get the age of the most recent entry in a channel's field feed

        `get-channel-field-feed field_last_data
        <https://mathworks.com/help/thingspeak/get-channel-field-feed.html#field_last_data>`_
        """
        if options is None:
            options = dict()
        if self.api_key is not None:
            options["api_key"] = self.api_key
        url = "{server_url}/channels/{id}/fields/{field}/last{fmt}".format(
            server_url=self.server_url, id=self.id, field=field, fmt=self.fmt
        )
        r = requests.get(url, params=options, timeout=self.timeout)
        return self._fmt(r)
    
channel = thingspeak.Channel(id=channel_id,api_key= read_key)
real_temperature = get_field_last(channel,field =1,options=None)
real_humidity = get_field_last(channel,field= 2,options=None)

#real time temperature and humidity readings
real_temperature = re.findall('\d+\.\d+', real_temperature)
real_humidity = re.findall('\d+\.\d+', real_humidity)

##########################################################################
#you can use this value for making prediction of the reading 
#print(Ensembelling.predict(np.array(float(real_temperature[0])).reshape(-1,1)))
#########################################################################################

real_temperature = real_temperature[0]

#finding possible crops acording to readings
#buiding decision tree
if (real_temperature == 25):
    crops = "Cotton"
elif (real_temperature >= str(10) and real_temperature <= str(15)):
    crops = "Linen, Wheat, Onions, Chickpeas"
elif (real_temperature >= str(15) and real_temperature <= str(20)):
    crops = "Grapes, Appels, orange, Strawberry, cabbage"
elif (real_temperature >= str(20) and real_temperature <= str(25)):
    crops = "Beet, pepper, Tomato, sunflower"
elif (real_temperature >= str(25) and real_temperature <= str(30)):
    crops = "cucumber, Eggplant,Beans, Sugar cane"
elif (real_temperature >= str(30) and ral_temperature <= str(35)):
    crops = "Soybean, Rice"

########################################################################
#Building the GUI to show results
print(crops)
root = tk.Tk()

result_show = tk.Canvas(root, width= 950, height = 800)
result_show.pack()

#createGUI label
Label_1 = tk.Label(root,text = "Green the AI Assistant",font =("Verdana",15))
result_show.create_window(475, 50, window=Label_1)


#show predictions results
Label_2 = tk.Label(root,text= "Error Percent :",font =("Italic",10))
result_show.create_window(320, 625, window=Label_2)

Label_3 = tk.Label(root,text = round(error_percent,4) ,font =("Italic",10))
result_show.create_window(390, 625, window=Label_3)

Label_4 = tk.Label(root,text= "Success Percent :",font =("Italic",10))
result_show.create_window(320, 600, window=Label_4)

Label_5 = tk.Label(root,text = round(sucess_percent,4)*100 ,font =("Italic",10))
result_show.create_window(390, 600, window=Label_5)

Label_8 = tk.Label(root,text = "Predection results Graph" ,font =("Verdana",12),fg = "blue")
result_show.create_window(458, 250, window=Label_8)

Label_11 = tk.Label(root,text= "Temperature now :",font =("Italic",10))
result_show.create_window(620, 600, window=Label_11)

Label_12 = tk.Label(root,text= real_temperature ,font =("Italic",10))
result_show.create_window(695, 600, window=Label_12)

img = ImageTk.PhotoImage(Image.open("logo22.ppm"))  
result_show.create_image(80, 70, anchor=NW, image=img)

img2 = ImageTk.PhotoImage(Image.open("logo11.ppm"))  
result_show.create_image(700, 70, anchor=NW, image=img2)

img3 = ImageTk.PhotoImage(Image.open("predictions result.ppm"))  
result_show.create_image(260, 280, anchor=NW, image=img3)

Label_6 = tk.Label(root,text = "According to predictions and weather forecasting you may plant: ",font =("Verdana",15))
result_show.create_window(475, 650, window=Label_6)

Label_7 = tk.Label(root,text = crops ,font =("Helvetica",12))
result_show.create_window(390, 690, window=Label_7)
root.mainloop()


# In[ ]:





# In[ ]:




