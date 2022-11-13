# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 01:04:35 2022

@author: M_ene
"""

#%% Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

"""
ModuleNotFoundError: No module named 'tensorflow_core.estimator'
 Böyle bir hata verdi tensorflow ile tensorflow estimators sürümleri eşit olmalı
"""


#%% Importing the dataset
dataset = pd.read_csv('Churn_modelling.csv')
x = dataset.iloc[:, 3:-1].values #ilk 3 coumns gereksiz o yüzden sağ tarafa bu sefer range belirtmeliyiz
y = dataset.iloc[:, -1].values # : kullanayarak sadece tek column seçebiliyoeuz çünkü bir range belirmiyoruz


#%%Şimdi String olan değerleri sayısal değerlere çevirelim

#Gender female=0,male=1 kendisi seçiyor
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])



#Ülkeler için

#france 1 0 0
#spain 0 0 1
#germany 0 1 0
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


#%%Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


#%% Feature Scaling
#neural network için hepsini scale etmeliyiz dedik

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



#%% Initializing the ANN
ann = tf.keras.models.Sequential()  # bu kısımdaki kodlar from import vs dite de ekleyebiliyorduk unutma

# Adding the input layer and the first hidden layer
ann.add(tf.keras.Input(shape=(12,))) #input
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) #first hidden layer
        
"""
add demek yerine direkt içine de yazabilirdik ama bu daha güzel
input shape de belirtebilrdik yani kaç tane input olduğunu, ama otomatik belirlenir dedik
ann.add(tf.keras.layers.Dense(units=6, activation='relu') , input_shape=(12,)) 
-ben ayrı olarak vermeyi tercih edicem

"""
# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer bir tane output var
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #eğer kategorik output olsaydı yani 2 den fazla değer içeren binary olmayan p zaman fonsiyonu softmax seçerdik

#%%


"""
The main difference between them is that the output variable in regression is numerical (or continuous) 
while that for classification is categorical (or discrete).
"""

"""
ek bilgi binary demek 1 ve 0 olarak dağıtılabilen columnslardır.
yani 2 farklı özellik eğer 2 den fazla olura bu kategorik olur

"""

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""
-burada binary bir tahmin yapmaya çalıştığımız için loss = 'binary_crossentropy
eğer kategorik olsaydı categorical_cros.... olurdu hatta output layer ise sigmoid değil softmax olurdu
- optimizer weight lerin  belirlenmesi için kullanılan yöntem

"""

# Training the ANN on the Training set
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)

#%% Making the predictions and evulating the model
"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

"""
Therefore, our ANN model predicts that this customer stays in the bank!
Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
- i.ine verdiğimiz değerleri scale ettik çünkü öyle çalışıyordu bir kere fit ettiğimiz için bir daha fit demeye gerek yok direkt transform yapabiliriz
"""

# Predicting the Test set results
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5) #threshhold yaptık

predict=np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
print(predict)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

