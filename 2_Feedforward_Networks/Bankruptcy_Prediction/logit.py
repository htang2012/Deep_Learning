import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

#data = pd.read_csv("/Users/jamesledoux/Documents/Research/deep_learning_book_notes/Data/LoanStats3b.csv")
data = pd.read_csv("../../Data/LoanStats3b.csv")
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

temp = pd.get_dummies(data['term'])
data['term'] = temp[' 36 months']

grades = pd.get_dummies(data['grade'])
data = pd.concat([data, grades], axis=1)

grades = pd.get_dummies(data['sub_grade'])
data = pd.concat([data, grades], axis=1)

pymnt_plan = pd.get_dummies(data['pymnt_plan'])
data = pd.concat([data, pymnt_plan], axis=1)

data['int_rate'] = data['int_rate'].str.strip('%').astype(float)

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
#for i in list(data):
#	data[i] = pd.to_numeric(data[i])

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


model = LogisticRegression(n_jobs=3)#, solver='liblinear')
model.fit(train_x, train_y)
preds = model.predict(test_x)
confusion_matrix = confusion_matrix(test_y, preds)
print confusion_matrix

