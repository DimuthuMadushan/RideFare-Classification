
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

dataset = pd.read_csv('train.csv')
imputer = SimpleImputer(missing_values =np.nan, strategy ='mean')
X = dataset[['duration','meter_waiting','meter_waiting_fare','meter_waiting_till_pickup','pick_lat','pick_lon','drop_lat','drop_lon','fare']]
Y = dataset['label']
Y = np.where(Y== 'correct',1,0)
imputer.fit(X)
X = imputer.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.75, max_features=2, max_depth=2, random_state=0)
gb_clf.fit(X_train, y_train)

gb_clf.fit(X_train, y_train)
predictions = gb_clf.predict(X_test)



print("Classification Report")
print(classification_report(y_test, predictions))    
test_dataset = pd.read_csv('test.csv')
test_df = test_dataset[['duration','meter_waiting','meter_waiting_fare','meter_waiting_till_pickup','pick_lat','pick_lon','drop_lat','drop_lon','fare']]
imputer.fit(test_df)
test_df = imputer.transform(test_df)
y_pred=gb_clf.predict(test_df)
Y_pred_dataframe = pd.DataFrame(y_pred)
Test_ids = test_dataset.iloc[:,0:1]
Y_test_ids_dataframe = pd.DataFrame(Test_ids)
