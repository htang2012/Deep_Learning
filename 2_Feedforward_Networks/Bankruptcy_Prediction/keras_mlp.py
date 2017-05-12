import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
import os

seed = 0

path = '/Users/jamesledoux/Downloads/Dane/1year.csv'
data = pd.read_csv(path)

#shuffle rows 
data = data.reindex(np.random.permutation(data.index)) #shuffle rows


#loan_status
y_data = data['class']

data = data.drop('class', 1)
#ensure everything is numeric
#for i in list(data):
#	data[i] = pd.to_numeric(data[i])

#mean-impute missing values
data = data.fillna(data.mean())
#data = data.fillna(0)

#data = data.reindex(np.random.permutation(data.index)) #shuffle rows
#y_data = y_data.reindex(np.random.permutation(y_data.index)) 
print("shape: ")
print(data.shape)

data.select_dtypes(include=['object'])

#data = pd.DataFrame(StandardScaler().fit_transform(data))

cutoff_val = int(np.floor(data.shape[0]*.8))
train_x = data[0:cutoff_val]
test_x = data[cutoff_val:data.shape[0]]

train_y= y_data[0:cutoff_val]
test_y = y_data[cutoff_val:y_data.shape[0]]

"""
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

train_x = X_train.reshape(60000, 784)
test_x = X_test.reshape(10000, 784)
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_y = pd.get_dummies(y_train)
test_y = pd.get_dummies(y_test)
"""

# create model
"""
model = Sequential()
model.add(Dense(512, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(512, input_dim=512, init='normal', activation='relu'))
model.add(Dense(512, input_dim=512, init='normal', activation='relu'))
model.add(Dense(512, input_dim=512, init='normal', activation='relu'))
model.add(Dense(256, input_dim=512, init='normal', activation='relu'))
model.add(Dense(256, input_dim=256, init='normal', activation='relu'))
model.add(Dense(256, input_dim=256, init='normal', activation='relu'))
model.add(Dense(32, input_dim=256, init='normal', activation='relu'))
model.add(Dense(1, input_dim=32, init='normal', activation='sigmoid'))
"""
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(32, input_dim=64, init='normal', activation='relu'))
model.add(Dense(1, input_dim=32, init='normal', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#encoder = LabelEncoder()
#encoder.fit(train_y)
#encoded_Y = encoder.transform(train_y)
#encoded_Y = np.reshape(encoded_Y, (encoded_Y.shape[0], 1))

estimator = model

train_x = np.matrix(train_x)
train_y = np.asmatrix(train_y).T
test_x = np.matrix(test_x)
test_y = np.asmatrix(test_y).T
train_x.shape
train_y.shape
history = estimator.fit(train_x, train_y, validation_split=0.20, nb_epoch=50, batch_size=64, verbose=1)

preds = model.predict(test_x)
preds = np.round(preds)
#confusion_matrix = confusion_matrix(test_y, preds)
print "mlp out of sample score: "
print confusion_matrix(test_y, preds)



model = LogisticRegressionCV(cv=5, n_jobs=3, penalty='l1', solver='liblinear')
model.fit(train_x, train_y)
preds = model.predict(test_x)
confusion_matrix = confusion_matrix(test_y, preds)
print confusion_matrix


model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


