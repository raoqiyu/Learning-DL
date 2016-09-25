from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.callbacks import EarlyStopping
def build_model():
    # begin
    model = Sequential()
    
    # 1st conv layer
    model.add(Convolution2D(32, 3, 3, border_mode='valid',input_shape=(1, 96, 96)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # 2nd conv layer
    model.add(Convolution2D(64, 2, 2, border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # 3rd conv layer
    model.add(Convolution2D(128, 2, 2, border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
   
    # 1st densely layer

    model.add(Flatten())
    model.add(Dense(500, init='uniform'))
    model.add(Activation('tanh'))
    
    # 2nd densely layer
    model.add(Dense(500, init='normal'))
    model.add(Activation('tanh'))
    
    # output layer
    model.add(Dense(30, init='normal'))
    model.add(Activation('tanh'))
    
    
    # Compile model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    
    return model
	