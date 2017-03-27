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

#data = pd.read_csv("/Users/jamesledoux/Documents/Research/deep_learning_book_notes/Data/LoanStats3b.csv")
data = pd.read_csv("LoanStats3b.csv")

data = data[data.loan_status != 'Late (16-30 days)']
data = data[data.loan_status != 'Late (31-120 days)']
data = data[data.loan_status != 'In Grace Period']
data = data[data.loan_status != 'Current']

data['target'] = 0
data.ix[data.loan_status == 'Default', 'target'] = 1
data.ix[data.loan_status == 'Charged Off', 'target'] = 1

data = data.drop('loan_status', 1)




categoricals_to_drop = ['term', 'grade', 'emp_title', 'home_ownership', 'verification_status', \
						'issue_d', 'url', 'desc', 'purpose', 'title', 'zip_code', 'addr_state', \
						'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d',\
						'emp_title', 'application_type', 'sub_grade', 'pymnt_plan', 'initial_list_status',\
						'application_type']


#scp LoanStats3b.csv ledouxja@136.167.92.85:/home/ledouxja

temp = pd.get_dummies(data['term'])
data['term'] = temp[' 36 months']

grades = pd.get_dummies(data['grade'])
data = pd.concat([data, grades], axis=1)

grades = pd.get_dummies(data['sub_grade'])
data = pd.concat([data, grades], axis=1)

pymnt_plan = pd.get_dummies(data['pymnt_plan'])
data = pd.concat([data, pymnt_plan], axis=1)

data['int_rate'] = data['int_rate'].str.strip('%').astype(float)

#titles = pd.get_dummies(data['emp_title']) #115k unique values.. can't do much with this. 
#data = pd.concat([data, titles], axis=1)

data.replace('n/a', np.nan,inplace=True)
data.emp_length.fillna(value=0,inplace=True)
data['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
data['emp_length'] = data['emp_length'].astype(int)

ownership = pd.get_dummies(data['home_ownership'])
data = pd.concat([data, ownership], axis=1)

verification = pd.get_dummies(data['verification_status'])
data = pd.concat([data, verification], axis=1)

data['desc'] = data['desc'].str.len()


purpose = pd.get_dummies(data['purpose'])
data = pd.concat([data, purpose], axis=1)


zip_code = pd.get_dummies(data['zip_code'])
data = pd.concat([data, zip_code], axis=1)


addr_state = pd.get_dummies(data['addr_state'])
data = pd.concat([data, addr_state], axis=1)

data['revol_util'] = data['revol_util'].str.strip('%').astype(float)

initial_list_status = pd.get_dummies(data['initial_list_status'])
data = pd.concat([data, initial_list_status], axis=1)


application_type = pd.get_dummies(data['application_type'])
data = pd.concat([data, application_type], axis=1)


for col in categoricals_to_drop:
	try:
		data = data.drop(str(col), 1)
	except:
		pass

#data.select_dtypes(include=['object'])

#shuffle rows 
data = data.reindex(np.random.permutation(data.index)) #shuffle rows


#loan_status
y_data = data['target']

data = data.drop('target', 1)
#ensure everything is numeric
for i in list(data):
	data[i] = pd.to_numeric(data[i])

#mean-impute missing values
data = data.fillna(data.mean())
data = data.fillna(0)

#data = data.reindex(np.random.permutation(data.index)) #shuffle rows
#y_data = y_data.reindex(np.random.permutation(y_data.index)) 
print(data.shape)

data.select_dtypes(include=['object'])

data = pd.DataFrame(StandardScaler().fit_transform(data))

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
model = Sequential()
model.add(Dense(512, input_dim=train_x.shape[1], init='normal', activation='relu'))
model.add(Dense(512, input_dim=512, init='normal', activation='relu'))
model.add(Dense(512, input_dim=512, init='normal', activation='relu'))
model.add(Dense(512, input_dim=512, init='normal', activation='relu'))
model.add(Dense(32, input_dim=512, init='normal', activation='relu'))
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
history = estimator.fit(train_x, train_y, validation_split=0.20, nb_epoch=10, batch_size=64, verbose=1)

preds = model.predict(test_x)
preds = np.round(preds)
confusion_matrix = confusion_matrix(test_y, preds)
print "mlp out of sample score: "
print confusion_matrix



model = LogisticRegressionCV(cv=5, n_jobs=3, penalty='l1', solver='liblinear')
model.fit(train_x, train_y)
preds = model.predict(test_x)
confusion_matrix = confusion_matrix(test_y, preds)
print confusion_matrix


"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

num_epochs = 1000
batch_size = 50
lr = 0.01


#load in data for 5, 4, 3, 2, and 1-year out bankruptcy prediction
dataframes = []
for f in os.listdir('csv_data'):
	f = 'csv_data/' + f
	print(f)
	df = pd.read_csv(f)
	dataframes.append(df)

df = dataframes[1]  # create a function for getting next_batch() of train/test data
df = df.reindex(np.random.permutation(df.index)) #shuffle rows
df = df.fillna(df.mean())
print(df.shape)
train, test = train_test_split(df, test_size = 0.3)
train_y = train['64']
train_y = train_y.reshape(train.shape[0], 1)
train_x = train.drop('64', 1)
test_y = test['64']
test_y = test_y.reshape(test.shape[0], 1)
test_x = test.drop('64', 1)
train_instances = train_x.shape[0]

#define variables used in the graph
X = tf.placeholder(tf.float32, [None, 65])
y = tf.placeholder(tf.float32, [None, 1]) 

w_h = tf.Variable(tf.random_normal([65,256], stddev=0.01))
b1 = tf.Variable(tf.zeros([256]))
w_h2 = tf.Variable(tf.random_normal([256,256], stddev=0.01))
b2 = tf.Variable(tf.zeros([256]))
w_out = tf.Variable(tf.random_normal([256,1], stddev=0.01))
b3 = tf.Variable(tf.zeros([1]))


#build the graph
h = tf.nn.relu(tf.matmul(X,w_h) + b1)
#h = tf.nn.dropout(h, .2)
h2 = tf.nn.relu(tf.matmul(h,w_h2) + b2)
#h2 = tf.nn.dropout(h2, .2)
output = tf.matmul(h2, w_out) + b3

#feed forward through the model
yhat = tf.sigmoid(output)

#loss function
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat, y))
cross_entropy = -tf.reduce_sum(y * tf.log(yhat))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, y))

# % of correct answers found in batch
preds = tf.round(yhat)
is_correct = tf.equal(y, preds)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


#optimizer function for backprop + weight updates
optimizer = tf.train.GradientDescentOptimizer(lr)
train_step = optimizer.minimize(loss)

predict_op = preds

init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(num_epochs):
		for batch in range(train_instances / batch_size):
			batch_x = train_x[batch * batch_size : (batch+1) * batch_size]
			batch_y = train_y[batch * batch_size : (batch+1) * batch_size]
			train = sess.run(train_step, feed_dict={X: batch_x, y: batch_y})
			a,c = sess.run([accuracy, loss], feed_dict={X: batch_x, y: batch_y})
			#print c
			#print sess.run(yhat,feed_dict={X: batch_x, y: batch_y})
			#print test set results
			#batch_xs, batch_ys = mnist.test.next_batch(batch_size)
			#acc = (np.mean(np.argmax(test_y, axis=1) == sess.run(predict_op, feed_dict={X: test_x})))
		a,c = sess.run([accuracy, loss], feed_dict={X: test_x, y: test_y})
		if epoch%10==0:
			print "Test set accuracy at epoch " + str(epoch) + ": " + str(a)
		if epoch % 200 == 0:
			print sess.run(yhat, feed_dict={X: test_x, y: test_y})
"""