
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
import datetime

dataset = pd.read_csv('train.csv')
imputer = SimpleImputer(missing_values =np.nan, strategy ='mean')

tempDur = dataset['duration']
PT = dataset['pickup_time']
DT = dataset['drop_time']
for i in range(len(tempDur)):
    if(pd.isnull(tempDur[i])):
        date0 = datetime.datetime.strptime(DT[i], "%m/%d/%Y %H:%M")
        date1 = datetime.datetime.strptime(PT[i], "%m/%d/%Y %H:%M")
        delta = date0 - date1
        tempDur[i] = delta.seconds
        
tempMW = dataset['meter_waiting']
for i in range(len(tempMW)):
    if(pd.isnull(tempMW[i])):
        tempMW[i]=0.0

tempLabel = dataset['label']
tempFare = dataset['fare']
for i in range(len(tempFare)):
    if(pd.isnull(tempFare[i]) or tempFare[i]==0 or tempFare[i]<0):
        tempFare[i]=0.0
        tempLabel[i]='incorrect'
        
dataset['duration'] = tempDur
dataset['label'] = tempLabel
dataset['fare'] = tempFare
dataset['duration'] = dataset['duration']-tempMW
X = dataset[['duration','meter_waiting','meter_waiting_fare','meter_waiting_till_pickup','pick_lat','pick_lon','drop_lat','drop_lon','fare']]
Y = dataset['label']
imputer.fit(X)
X = imputer.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=200,max_features='sqrt',max_depth=40)
clf.fit(X_train,y_train)

test_dataset = pd.read_csv('test.csv')
test_df = test_dataset[['duration','meter_waiting','meter_waiting_fare','meter_waiting_till_pickup','pick_lat','pick_lon','drop_lat','drop_lon','fare']]
imputer.fit(test_df)
test_df = imputer.transform(test_df)
y_pred=clf.predict(test_df)
y_pred =np.where(y_pred == 'correct',1,0)
Y_pred_dataframe = pd.DataFrame(y_pred)
Test_ids = test_dataset.iloc[:,0:1]
Y_test_ids_dataframe = pd.DataFrame(Test_ids)


