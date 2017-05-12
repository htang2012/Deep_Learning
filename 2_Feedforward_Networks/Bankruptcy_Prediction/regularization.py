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

act = 'tanh'
path = '/Users/jamesledoux/Downloads/Dane/1year.csv'
data = pd.read_csv(path)
data = data.drop('Unnamed: 0', 1) #this is meaningless

#shuffle rows 
seed = 0
data = data.reindex(np.random.permutation(data.index)) #shuffle rows
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

"""
dropout_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
accuracies = []
for val in dropout_vals:
	print "training model with dropout value {}".format(val)
	model = Sequential()
	model.add(Dense(1024, input_dim=train_x.shape[1], init='normal', activation=act))
	model.add(Dropout(val))
	model.add(Dense(1024, input_dim=1024, init='normal', activation=act))
	model.add(Dropout(val))
	model.add(Dense(1024, input_dim=1024, init='normal', activation=act))
	model.add(Dropout(val))
	model.add(Dense(1, input_dim=1024, init='normal', activation='sigmoid'))
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
"""


reg_constants = [0.0, 0.001, 0.01, 0.1, 0.5, 0.9]
accuracies = []
for reg_constant in reg_constants:
	print "training model with dropout value {}".format(val)
	model = Sequential()
	model.add(Dense(1024, input_dim=train_x.shape[1], init='normal', activation=act, W_regularizer=l1(reg_constant)))
	model.add(Dropout(val))
	model.add(Dense(1024, input_dim=1024, init='normal', activation=act, W_regularizer=l1(reg_constant)))
	model.add(Dropout(val))
	model.add(Dense(1024, input_dim=1024, init='normal', activation=act, W_regularizer=l1(reg_constant)))
	model.add(Dropout(val))
	model.add(Dense(1, input_dim=1024, init='normal', activation='sigmoid', W_regularizer=l1(reg_constant)))
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



"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.regularizers import l1
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

path = '/Users/jamesledoux/Downloads/Dane/1year.csv'
data = pd.read_csv(path)
data = data.drop('Unnamed: 0', 1) #this is meaningless

# look at distributions
#for i in data.columns[1:5]:
#	data[i].hist(bins=20)
#	plt.show()

#look at summary stats
#data.describe()

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
	#print '''{}: 
	#lower outlier threshold: {}, 
	#upper outlier threshold: {}'''.format(col, lower_outlier_threshold, upper_outlier_threshold)
	data.loc[data[col]>upper_outlier_threshold, col] = upper_outlier_threshold
	data.loc[data[col]<lower_outlier_threshold, col] = lower_outlier_threshold

#mean-impute missing values
data = data.fillna(data.mean())


#revisit distributions
#for i in data.columns[1:5] #again, limit this for sanity's sake. you get the picture after seeing a few. 
#	data[i].hist(bins=20)
#	plt.show()

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


dropout_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
accuracies = []
for dropout_val in dropout_vals:
	print "training single-layer model with dropout value {}".format(dropout_val)
	model = Sequential()
	model.add(Dense(128, input_dim=train_x.shape[1], init='normal', activation='relu'))
	model.add(Dropout(dropout_val))
	model.add(Dense(64, input_dim=128, init='normal', activation='relu'))
	model.add(Dropout(dropout_val))
	model.add(Dense(32, input_dim=64, init='normal', activation='relu'))
	model.add(Dropout(dropout_val))
	model.add(Dense(1, input_dim=32, init='normal', activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

	weights_path = 'best_weights.hdf5'
	mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
	                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
	#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
	history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=10, 
		batch_size=64, verbose=0, callbacks=[mcp]) #orig. 200 epochs. change back when done testing w_reg

	preds = model.predict(test_x)
	preds = np.round(preds)
	#confusion_matrix = confusion_matrix(test_y, preds)
	print "mlp out of sample score: "
	cm = confusion_matrix(test_y, preds)
	preds = preds.ravel()
	auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
	print cm
	print auc#"auc: %d" %auc
	acc = (float(cm[0][0]) + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
	print acc

	#plot training history
	# plt.plot(history.history['acc'])
	# plt.plot(history.history['val_acc'])
	# plt.title('model accuracy')
	# plt.ylabel('accuracy')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper left')
	# plt.show()

	# # summarize history for loss
	# plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	# plt.title('model loss')
	# plt.ylabel('loss')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper left')
	# plt.show()



# reg_constants = [0, .0001, .0005, 0.001, .003]
# accuracies = []
# for reg_constant in reg_constants:
# 	print "L2 Weight Regularization Constant: {}".format(reg_constant)
# 	model = Sequential()
# 	model.add(Dense(128, input_dim=train_x.shape[1], init='normal', activation='relu', W_regularizer=l2(reg_constant)))
# 	model.add(Dense(64, input_dim=128, init='normal', activation='relu', W_regularizer=l2(reg_constant)))
# 	model.add(Dense(32, input_dim=64, init='normal', activation='relu', W_regularizer=l2(reg_constant)))
# 	model.add(Dense(1, input_dim=32, init='normal', activation='sigmoid', W_regularizer=l2(reg_constant)))
# 	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

# 	weights_path = 'best_weights.hdf5'
# 	mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
# 	                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
# 	#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
# 	history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=300, 
# 		batch_size=64, verbose=0, callbacks=[mcp])

# 	preds = model.predict(test_x)
# 	preds = np.round(preds)
# 	#confusion_matrix = confusion_matrix(test_y, preds)
# 	print "mlp out of sample score: "
# 	cm = confusion_matrix(test_y, preds)
# 	preds = preds.ravel()
# 	auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
# 	print cm
# 	print auc#"auc: %d" %auc
# 	acc = (float(cm[0][0]) + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
# 	print acc

# 	# plot training history
# 	plt.plot(history.history['acc'])
# 	plt.plot(history.history['val_acc'])
# 	plt.title('model accuracy')
# 	plt.ylabel('accuracy')
# 	plt.xlabel('epoch')
# 	plt.legend(['train', 'test'], loc='upper left')
# 	plt.show()

# 	# summarize history for loss
# 	plt.plot(history.history['loss'])
# 	plt.plot(history.history['val_loss'])
# 	plt.title('model loss')
# 	plt.ylabel('loss')
# 	plt.xlabel('epoch')
# 	plt.legend(['train', 'test'], loc='upper left')
# 	plt.show()


# next: L1, then repeat for activation regs.

reg_constants = [.0001, .0005, 0.001]
accuracies = []
for reg_constant in reg_constants:
	print "L1 Weight Regularization Constant: {}".format(reg_constant)
	model = Sequential()
	model.add(Dense(128, input_dim=train_x.shape[1], init='normal', activation='relu', W_regularizer=l1(reg_constant)))
	model.add(Dense(64, input_dim=128, init='normal', activation='relu', W_regularizer=l1(reg_constant)))
	model.add(Dense(32, input_dim=64, init='normal', activation='relu', W_regularizer=l1(reg_constant)))
	model.add(Dense(1, input_dim=32, init='normal', activation='sigmoid', W_regularizer=l1(reg_constant)))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile model

	weights_path = 'best_weights.hdf5'
	mcp = ModelCheckpoint(weights_path, monitor="val_acc", mode='max',
	                  save_best_only=True, save_weights_only=False, verbose=0) #verbose=1 is a good way to verify this is working
	#callbacks_list = [EarlyStopping(monitor='binary_crossentropy', patience=15)]
	history = model.fit(train_x, train_y, validation_split=0.20, nb_epoch=300, 
		batch_size=64, verbose=0, callbacks=[mcp])

	preds = model.predict(test_x)
	preds = np.round(preds)
	#confusion_matrix = confusion_matrix(test_y, preds)
	print "mlp out of sample score: "
	cm = confusion_matrix(test_y, preds)
	preds = preds.ravel()
	auc = roc_auc_score(test_y.reshape(-1).tolist()[0], preds.tolist())
	print cm
	print auc#"auc: %d" %auc
	acc = (float(cm[0][0]) + cm[1][1])/(cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
	print acc

	# plot training history
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

"""