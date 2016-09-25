import os
import numpy as np
import pandas as pd


def save(y, fname):

	# get feature names
	df = pd.read_csv('./data/training.csv')
	feature_names = df.columns[:30].values
	n_features = len(feature_names)
	print(feature_names)
	# number of images in the test set
	n_imgs = 1783
	
	# define the prediction matrix
	# 4 columns : Row Id, Image Id, Feature Name, Location
	
	# Fill Row ID
	RowID  = [i for i in range(1,n_imgs*n_features+1)]
	# Fill Image ID
	ImageId = np.asarray([ [i]*n_features for i in range(1,n_imgs+1)]).flatten()
	# Fill Feature Name
	FeatureName = np.asarray([ feature_names for i in range(n_imgs)]).flatten()
	# Fill Location
	Location  = y.flatten()
	
	df = pd.DataFrame({'ImageId' : ImageId, 'FeatureName' : FeatureName,'Location':Location}, RowID)
	
	df.to_csv(fname,index_label=['RowId'])