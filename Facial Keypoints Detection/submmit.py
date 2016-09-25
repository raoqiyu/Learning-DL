import os
import numpy as np
import pandas as pd


def save(y, fname):

	# get feature names, index
	df = pd.read_csv('./data/training.csv')
	feature_names = df.columns[:30].values
	n_features = len(feature_names)
	feature_index = [i for i in range(n_features)]
	name_2_idx = {}
	for i, n in enumerate(feature_names):
		name_2_idx[n] = i
		
	print(feature_names)


	# number of images in the test set
	n_imgs = 1783

	# load id 
	df = pd.read_csv('./data/IdLookUpTable.csv')
	id_look_up = df[['ImageId','FeatureName']].values
	# number of all points should be predicted
	n_rows = df.shape[0]

	# y ; n_imgs x n_features

	# 1 columns, Location
	prediction_2_submit = np.zeros((n_rows, 1))
	for i in range(n_rows):
		prediction_2_submit[i] = y[id_look_up[i,0]-1, name_2_idx[id_look_up[i,1]]]

	# generate Row id
	RowID = [i for i in range(1,n_rows+1)]
		
		
	df = pd.DataFrame(prediction_2_submit, index=RowID, columns=['Location'])

	df.to_csv(fname,index_label=['RowId'])