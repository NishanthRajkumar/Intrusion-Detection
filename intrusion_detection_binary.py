import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# print(physical_devices)

train = pd.read_csv('Dataset/KDDTrain+.csv')

test = pd.read_csv('Dataset/KDDTest+.csv')

test2 = pd.read_csv('Dataset/KDDTest-21.csv')

train.head(5)

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

x_train = train.iloc[:, train.columns != 'class']
y_train = train['class']

x_test = test.iloc[:, test.columns != 'class']
y_test = test['class']

x_test2 = test2.iloc[:, test2.columns != 'class']
y_test2 = test2['class']

#Encodes the categorical columns
for column in x_train.columns:
    if x_train[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        x_train[column] = le.fit_transform(x_train[column])

for column in x_test.columns:
    if x_test[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        x_test[column] = le.fit_transform(x_test[column])

for column in x_test2.columns:
    if x_test2[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        x_test2[column] = le.fit_transform(x_test2[column])

# x_train.head(5)

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

for i in y_test2:
    if i in att.keys():
        if att[i] == 0:
            Y_test2.append(0)
        else:
            Y_test2.append(1)
    else:
        Y_test2.append(1)


Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)
Y_test2 = np.asarray(Y_test2)

#Initialize the no of hidden neurons, learning rate and epoch
hidden_neurons = 80
learning_rate = 0.001
ep = 30

classifier = Sequential()
classifier.add(Dense(2048, activation='relu', input_dim=41)) #Since we have 42 columns
classifier.add(Dense(1024, activation='relu'))
classifier.add(Dense(512, activation='relu'))
classifier.add(Dense(256, activation='relu'))
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(32, activation='relu'))
classifier.add(Dense(16, activation='relu'))
classifier.add(Dense(8, activation='relu'))
classifier.add(Dense(4, activation='relu'))
classifier.add(Dense(2, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))
# rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# opt = optimizers.Adam(learning_rate=0.1)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])

# classifier.summary()

#Fitting the RNN to the training set
start = datetime.now()
keras_callback=[EarlyStopping(monitor='loss',patience=3,mode='min'), ModelCheckpoint('TL_Check_Point',monitor='loss',save_best_only=True)]
# a = classifier.fit(x_train, Y_train, epochs = ep, batch_size=512)
# a = classifier.fit(x_train, Y_train, epochs = ep, validation_data=(x_test, Y_test), callbacks=[keras_callback])
a = classifier.fit(x_train, Y_train, epochs = ep, validation_split = 0.10, callbacks=[keras_callback])
# a = classifier.fit(x_train, Y_train, epochs = ep, validation_split = 0.10)
train_time = datetime.now() - start

#Predicting the testing set
pred = classifier.predict(x_test)
pred1 = np.argmax(pred, axis=1)

pred = classifier.predict(x_test2)
pred2 = np.argmax(pred, axis=1)

#Print Accuracy
test_accuracy = accuracy_score(Y_test, pred1)
test2_accuracy = accuracy_score(Y_test2, pred2)
print('Test+ Accuracy: ', test_accuracy)
print('Test-21 Accuracy: ', test2_accuracy)

test_recall=recall_score(Y_test, pred1)
test2_recall=recall_score(Y_test2, pred2)
print('Test+ Recall: ', test_recall)
print('Test-21 Recall: ', test2_recall)

#Print Confusion Matrix
cm_test = confusion_matrix(Y_test, pred1)
cm_test2 = confusion_matrix(Y_test2, pred2)
print('Confusion Matrix for test+', cm_test)
print('Confusion Matrix for test-21', cm_test2)

# Evaluate classifier model for test+ data
print("Test+ dataset")
print(classifier.evaluate(x_test, Y_test))

# Evaluate classifier model for test-21 data
print("Test-21 dataset")
print(classifier.evaluate(x_test2, Y_test2))

#Print Time taken
print('time taken for train+: ', train_time)

plt.plot(a.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()