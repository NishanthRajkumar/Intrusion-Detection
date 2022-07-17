import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
# from tensorflow.keras.utils import np_utils
# from tensorflow.keras.utils import to_categorical

training_set = pd.read_csv('Dataset/KDDTrain+.csv')

train = training_set.iloc[:, :42]

testing_set = pd.read_csv('Dataset/KDDTest+.csv')
test = testing_set.iloc[:, :42]

testing_set2 = pd.read_csv('Dataset/KDDTest-21.csv')
test2 = testing_set2.iloc[:, :42]


att = {'normal':0,
       'back':1,
       'buffer_overflow':2,
       'ftp_write':3,
       'guess_passwd':3,
       'imap':3,
       'ipsweep':4,
       'land':1,
       'loadmodule':2,
       'multihop':3,
       'neptune':1,
       'nmap':4,
       'perl':2,
       'phf':3,
       'pod':1,
       'portsweep':4,
       'rootkit':2,
       'satan':4,
       'smurf':1,
       'spy':3,
       'teardrop':1,
       'warezclient':3,
       'warezmaster':3
    }

x_train = train.iloc[:  , :-1]
y_train = train.iloc[:,-1]

x_test = test.iloc[:  , :-1]
y_test = test.iloc[:,-1]

x_test2 = test.iloc[:  , :-1]
y_test2 = test.iloc[:,-1]


#Encodes the categorical columns
for column in x_train.columns:
    if x_train[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        x_train[column] = le.fit_transform(x_train[column])
X_train = x_train.iloc[:, :-1].values

for column in x_test.columns:
    if x_test[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        x_test[column] = le.fit_transform(x_test[column])
X_test = x_test.iloc[:, :-1].values

for column in x_test2.columns:
    if x_test2[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        x_test2[column] = le.fit_transform(x_test2[column])
X_test2 = x_test2.iloc[:, :-1].values



Y_train = []
Y_test = []
Y_test2 = []

#Assigning integers to represent the 22 types of attacks
for i in y_train:
    if i in att.keys():
        if att[i] == 0:
            Y_train.append(0)
        else:
            Y_train.append(1)
    else:
        Y_train.append(1)

for i in y_test:
    if i in att.keys():
        if att[i] == 0:
            Y_test.append(0)
        else:
            Y_test.append(1)
    else:
        Y_test.append(1)

for i in y_test:
    if i in att.keys():
        if att[i] == 0:
            Y_test2.append(0)
        else:
            Y_test2.append(1)
    else:
        Y_test2.append(1)


Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
Y_test2 = np.array(Y_test2)

'''Y_train = to_categorical(Y_train, 5)
sc = MinMaxScaler(feature_range = (0,1))
X_train = sc.fit_transform(X_train)'''

#Reshape the training features
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
X_test2 = np.reshape(X_test2, (X_test2.shape[0], 1, X_test2.shape[1]))

classifier = Sequential()

########################################################################
#Initialize the no of hidden neurons, learning rate and epoch
hidden_neurons = 80
learning_rate = 0.5
ep = 100            #Epochs
########################################################################

classifier.add(LSTM(units = hidden_neurons, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2]), activation = 'sigmoid', recurrent_activation='sigmoid'))
classifier.add(Dropout(0.2))
classifier.add(LSTM(units = hidden_neurons, return_sequences = True, activation = 'sigmoid', recurrent_activation='sigmoid'))
classifier.add(Dropout(0.2))
classifier.add(LSTM(units = hidden_neurons, return_sequences = True, activation = 'sigmoid', recurrent_activation='sigmoid'))
classifier.add(Dropout(0.2))
classifier.add(LSTM(units = hidden_neurons, activation = 'sigmoid', recurrent_activation='sigmoid'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, activation = 'softmax'))
classifier.add(Dropout(0.2))
rms = optimizers.RMSprop(lr=0.05, rho=0.9, epsilon=None, decay=0.0)
classifier.compile(optimizer = rms, loss = 'binary_crossentropy', metrics=['acc'])

#Fitting the RNN to the training set
start = datetime.now()
a = classifier.fit(X_train, Y_train, epochs = ep, validation_split = 0.05, shuffle = True)
train_time = datetime.now() - start

#Predicting the testing set
pred1 = classifier.predict(X_test)
pred2 = classifier.predict(X_test)
pred = classifier.predict(X_train)

#Print Accuracy
train_accuracy = accuracy_score(Y_train, pred)
test_accuracy = accuracy_score(Y_test, pred1)
test2_accuracy = accuracy_score(Y_test2, pred2)
print('Train+ Accuracy: ', train_accuracy)
print('Test+ Accuracy: ', test_accuracy)
print('Test-21 Accuracy: ', test2_accuracy)

#Print Confusion Matrix
cm_train = confusion_matrix(Y_train, pred)
cm_test = confusion_matrix(Y_test, pred1)
cm_test2 = confusion_matrix(Y_test, pred2)
print('Confusion Matrix for train+', cm_train)
print('Confusion Matrix for test+', cm_train)
print('Confusion Matrix for test-21', cm_train)

start = datetime.now()
b = classifier.fit(X_test, Y_test, epochs = ep, validation_split = 0.05, shuffle = True)
test_time = datetime.now() - start
start = datetime.now()
c = classifier.fit(X_test2, Y_test2, epochs = ep, validation_split = 0.05, shuffle = True)
test2_time = datetime.now() - start

#Print Time taken
print('time taken for train+: ', train_time)
print('time taken for test+: ', test_time)
print('time taken for test-21: ', test2_time)
tot_time = train_time + test_time + test2_time
print('Total Time: ', tot_time)


plt.plot(a.history['acc'])
plt.plot(b.history['acc'])
plt.plot(c.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train+', 'test+', 'test-21'], loc='upper right')
plt.show()