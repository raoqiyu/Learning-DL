import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def load(fname='train.csv', verbose=True):
    """
    load raw data into numpy array, 7049 training data, 1783 testing data
    data structure:
    training data:
        Each row contains the (x,y) coordinates for 15 keypoints, and image data as row-ordered list of pixels
    test data:
        Each row contains imageId and image data as row-ordered list of pixels
        
    Notice:
        scales all values to a smaller interval for easy computing
        scales the intensity values of the image pixels to be in the interval [0,1], instead of [0,255]
        scales the target values(x and y coordinates) are scaled to [-1,1], instead of [0,95]
    """
    
    df = pd.read_csv(fname)
    
    # convert Image column to numpy array
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    
    # print the number of values of each column
    if verbose:
        print(df.count())
    
    # drop all rows that have missing values
    df = df.dropna()
    
    # form training data, scale pixel values to [0,1]
    X = np.vstack(df['Image'].values)/255.
    X = X.astype(np.float32)
    
    # if the training data is loaded, it has target values
    if 'training' in fname:
        # target is the keypoints columns
        y = df[df.columns[:-1]].values
        # scale target coordinates to [-1,1]
        y = (y-48)/48
        # shuffle the train data
        X, y = shuffle(X, y)
        y = y.astype(np.float32)
    else:
        y = None
        
    return X, y
  
    
def load_2d(fname):
    """
    Using convolutional layer, the input should has the shape of (c, 0, 1), where c is the number of the number 
    of channels(colors), and 0 and 1 correspond to the x and y dimesions of the input image.
    In here, c == 1, and the dimensions of he image are 96 and 96
    """
    X, y = load(fname,verbose=False)
    X = X.reshape(-1, 1, 96, 96)
    return X, y
        
                                    
                                    