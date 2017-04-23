import numpy as np
import pandas as pd
import os

seed = 0

#convert arff to csv
def getCSVFromArff(fileNameArff):
    with open(fileNameArff + '.arff', 'r') as fin:
        data = fin.read().splitlines(True)
    i = 0
    cols = []
    for line in data:
        if ('@data' in line):
            i+= 1
            break
        else:
            #print line
            i+= 1
            if (line.startswith('@attribute')):
                if('{' in line):
                    cols.append(line[11:line.index('{')-1])
                else:
                    cols.append(line[11:line.index('numeric')-1])
    headers = ",".join(cols)
    with open(fileNameArff + '.csv', 'w') as fout:
        fout.write(headers)
        fout.write('\n')
        fout.writelines(data[i:])


def clean_data(df):
	"""
	dependent variable: 'class'
	others: Attr1, ..., Attr64
	"""
	df = df.replace('?', np.NaN) #NAs were originally recorded as ?'s
	#convert all columns to floats (some are strings), mean impute NAs
	for c in df.columns:
		df[c] = df[c].astype(float)
		df[c] = df[c].fillna(df[c].mean())
	return dfls


filenames = ['1year', '2year', '3year', '4year', '5year']
for f in filenames:
	getCSVFromArff(f)
	filename = f + '.csv'
	df = pd.read_csv(filename)
	df = clean_data(df)
	print f
	df.to_csv(filename)






#
