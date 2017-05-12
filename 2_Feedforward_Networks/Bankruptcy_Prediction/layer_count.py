from __future__ import division
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import os
from datetime import datetime


learning_rate=0.0001

path = '/Users/jamesledoux/Downloads/Dane/1year.csv'
data = pd.read_csv(path)
data = data.drop('Unnamed: 0', 1) #this is meaningless

#shuffle rows 
seed = 0
data = data.reindex(np.random.permutation(data.index)) #shuffle rows
#loan_status (NOTE: imoprtant to do this before removing outliers, 
# since this stp would remove the binary dependent variable for bankruptcy
# if this were still in the dataframe
y_data = data['class']
data = data.drop('class', 1)

#replace extreme outliers with upper quartile + 1.5 x IQR (Fijorek and Grotowski 2012)
for col in data.columns:  #limit to the first few for now. takes a long time to click through otherwise
	iqr = data[col].quantile(.75) - data[col].quantile(.25)
	upper_outlier_threshold = data[col].quantile(.75) + iqr*1.5
	lower_outlier_threshold = data[col].quantile(.25) - iqr*1.5
	data.loc[data[col]>upper_outlier_threshold, col] = upper_outlier_threshold
	data.loc[data[col]<lower_outlier_threshold, col] = lower_outlier_threshold

#mean-impute missing values
data = data.fillna(data.mean())

#scale to mean zero and unit variance 
data = pd.DataFrame(scale(data))
print("shape: ")
print(data.shape)

#data.select_dtypes(include=['object'])

#data = pd.DataFrame(StandardScaler().fit_transform(data))

cutoff_val = int(np.floor(data.shape[0]*.6))
train_x = data[0:cutoff_val]
test_x = data[cutoff_val:data.shape[0]]

train_y= y_data[0:cutoff_val]
test_y = y_data[cutoff_val:y_data.shape[0]]

train_x = np.matrix(train_x)
train_y = np.asmatrix(train_y).T
test_x = np.matrix(test_x)
test_y = np.asmatrix(test_y).T
train_x.shape
train_y.shape


# one layer
print "training model with %d hidden layer" % 1
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(1, input_dim=64, init='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

#checkpoint model to save best weights. load these for prediction
weights_path = 'best_weights.hdf5'
mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=200, 
	batch_size=64, verbose=0, callbacks=[mcp])#, callbacks=callbacks_list)

model.load_weights(weights_path) #load best weights from training
preds = model.predict(test_x)
preds = np.round(preds)
#confusion_matrix = confusion_matrix(test_y, preds)
print "mlp out of sample score: "
cm = confusion_matrix(test_y, preds)
preds = preds.ravel()
auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
print cm
print 'auc: {}'.format(auc)#"auc: %d" %auc
acc = (cm[0][0] + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
print acc

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#two layers
print "training model with %d hidden layer" % 2
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(1, input_dim=64, init='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

#checkpoint model to save best weights. load these for prediction
weights_path = 'best_weights.hdf5'
mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=200, 
	batch_size=64, verbose=0, callbacks=[mcp])#, callbacks=callbacks_list)

model.load_weights(weights_path) #load best weights from training
preds = model.predict(test_x)
preds = np.round(preds)
#confusion_matrix = confusion_matrix(test_y, preds)
print "mlp out of sample score: "
cm = confusion_matrix(test_y, preds)
preds = preds.ravel()
auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
print cm
print 'auc: {}'.format(auc)#"auc: %d" %auc
acc = (cm[0][0] + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
print acc

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



#three layers
print "training model with %d hidden layer" % 3
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(1, input_dim=64, init='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

#checkpoint model to save best weights. load these for prediction
weights_path = 'best_weights.hdf5'
mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=200, 
	batch_size=64, verbose=0, callbacks=[mcp])#, callbacks=callbacks_list)

model.load_weights(weights_path) #load best weights from training
preds = model.predict(test_x)
preds = np.round(preds)
#confusion_matrix = confusion_matrix(test_y, preds)
print "mlp out of sample score: "
cm = confusion_matrix(test_y, preds)
preds = preds.ravel()
auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
print cm
print 'auc: {}'.format(auc)#"auc: %d" %auc
acc = (cm[0][0] + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
print acc

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#four layers
print "training model with %d hidden layer" % 4
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(1, input_dim=64, init='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

#checkpoint model to save best weights. load these for prediction
weights_path = 'best_weights.hdf5'
mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=200, 
	batch_size=64, verbose=0, callbacks=[mcp])#, callbacks=callbacks_list)

model.load_weights(weights_path) #load best weights from training
preds = model.predict(test_x)
preds = np.round(preds)
#confusion_matrix = confusion_matrix(test_y, preds)
print "mlp out of sample score: "
cm = confusion_matrix(test_y, preds)
preds = preds.ravel()
auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
print cm
print 'auc: {}'.format(auc)#"auc: %d" %auc
acc = (cm[0][0] + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
print acc
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#five layers
print "training model with %d hidden layer" % 5
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(1, input_dim=64, init='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

#checkpoint model to save best weights. load these for prediction
weights_path = 'best_weights.hdf5'
mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=200, 
	batch_size=64, verbose=0, callbacks=[mcp])#, callbacks=callbacks_list)

model.load_weights(weights_path) #load best weights from training
preds = model.predict(test_x)
preds = np.round(preds)
#confusion_matrix = confusion_matrix(test_y, preds)
print "mlp out of sample score: "
cm = confusion_matrix(test_y, preds)
preds = preds.ravel()
auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
print cm
print 'auc: {}'.format(auc)#"auc: %d" %auc
acc = (cm[0][0] + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
print acc
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#six layers
print "training model with %d hidden layer" % 6
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(1, input_dim=64, init='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

#checkpoint model to save best weights. load these for prediction
weights_path = 'best_weights.hdf5'
mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=200, 
	batch_size=64, verbose=0, callbacks=[mcp])#, callbacks=callbacks_list)

model.load_weights(weights_path) #load best weights from training
preds = model.predict(test_x)
preds = np.round(preds)
#confusion_matrix = confusion_matrix(test_y, preds)
print "mlp out of sample score: "
cm = confusion_matrix(test_y, preds)
preds = preds.ravel()
auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
print cm
print 'auc: {}'.format(auc)#"auc: %d" %auc
acc = (cm[0][0] + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
print acc
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#7
print "training model with %d hidden layer" % 6
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(1, input_dim=64, init='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

#checkpoint model to save best weights. load these for prediction
weights_path = 'best_weights.hdf5'
mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=200, 
	batch_size=64, verbose=0, callbacks=[mcp])#, callbacks=callbacks_list)

model.load_weights(weights_path) #load best weights from training
preds = model.predict(test_x)
preds = np.round(preds)
#confusion_matrix = confusion_matrix(test_y, preds)
print "mlp out of sample score: "
cm = confusion_matrix(test_y, preds)
preds = preds.ravel()
auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
print cm
print 'auc: {}'.format(auc)#"auc: %d" %auc
acc = (cm[0][0] + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
print acc
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#8
print "training model with %d hidden layer" % 6
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(1, input_dim=64, init='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

#checkpoint model to save best weights. load these for prediction
weights_path = 'best_weights.hdf5'
mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=200, 
	batch_size=64, verbose=0, callbacks=[mcp])#, callbacks=callbacks_list)

model.load_weights(weights_path) #load best weights from training
preds = model.predict(test_x)
preds = np.round(preds)
#confusion_matrix = confusion_matrix(test_y, preds)
print "mlp out of sample score: "
cm = confusion_matrix(test_y, preds)
preds = preds.ravel()
auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
print cm
print 'auc: {}'.format(auc)#"auc: %d" %auc
acc = (cm[0][0] + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
print acc
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#9
print "training model with %d hidden layer" % 6
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(1, input_dim=64, init='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

#checkpoint model to save best weights. load these for prediction
weights_path = 'best_weights.hdf5'
mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=200, 
	batch_size=64, verbose=0, callbacks=[mcp])#, callbacks=callbacks_list)

model.load_weights(weights_path) #load best weights from training
preds = model.predict(test_x)
preds = np.round(preds)
#confusion_matrix = confusion_matrix(test_y, preds)
print "mlp out of sample score: "
cm = confusion_matrix(test_y, preds)
preds = preds.ravel()
auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
print cm
print 'auc: {}'.format(auc)#"auc: %d" %auc
acc = (cm[0][0] + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
print acc
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




#10 layers 
print "training model with %d hidden layer" % 6
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(64, input_dim=64, init='normal', activation='relu'))
model.add(Dense(1, input_dim=64, init='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

#checkpoint model to save best weights. load these for prediction
weights_path = 'best_weights.hdf5'
mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=200, 
	batch_size=64, verbose=0, callbacks=[mcp])#, callbacks=callbacks_list)

model.load_weights(weights_path) #load best weights from training
preds = model.predict(test_x)
preds = np.round(preds)
#confusion_matrix = confusion_matrix(test_y, preds)
print "mlp out of sample score: "
cm = confusion_matrix(test_y, preds)
preds = preds.ravel()
auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
print cm
print 'auc: {}'.format(auc)#"auc: %d" %auc
acc = (cm[0][0] + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
print acc
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


