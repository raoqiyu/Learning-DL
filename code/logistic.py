# -*- coding: utf-8 -*-
"""
Logistic Regression

From 
Deep Learning Tutorial

function load_data has beed modified to just load data locally

@Wahah
"""

import theano
import theano.tensor as T
import numpy

import timeit
import pickle
import os
import gzip
#%%
class LogisticRegression(object):
    """Multi-Class Logistic Regression Class
    
    The Logistic regression is fully described by a weight matrix W and bias vector b.
    Classification is done by projecting data points onto a set of hyperplanes, the 
    distance to which is used to determine a class membership probability.
    """
    
    def __init__(self, input, n_in, n_out):
        """
        :param input: symbolic variable that describe the input of the 
                        architecture (one minibatch)
        :param n_in:  number of input units, the dimension of the space in 
                        which the datapoints lie.
        :param n_out: number of output units, the dimension of the space in 
                        which the labels lie.
        """
        
        self.W = theano.shared(
            value=numpy.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
            )
        
        self.b = theano.shared(
            value=numpy.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        theano
        
        # symbolic expression for computing for computing the martix of the class-membership
        # probabilites
        # Where:
        # W is a matix where column-k represent the separation hyperplane for class-k
        # x is a matrix where row-j represents input training sample-j
        # b is a vector where element-k represent the free paramter of hyperplane-k
        self.p_y_given = T.nnet.softmax(T.dot(input, self.W) + self.b)
        
        # synbolic description of how to compute prediction as class whose probablity is maximal
        self.y_pred = T.argmax(self.p_y_given, axis=1)
        
        # paramter of the model
        self.params = [self.W, self.b]
        
        # keep track of model input
        self.input = input
        
    def negative_log_likelihood(self,y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution        
        """
        
        # y.shape[0] is(symbolically) the number of rows in y, i.e.,
        #   number of example (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,...,n-1]
        # T.log(self.p_y_given_x) is a matix of the Log-Probabilities (call it LP) with
        #   one row per example and one column per class
        # LP[T.arange(y.shape[0],y)] is a vector v containing [LP[0,y[0]], LP[1,y[1]],...]
        # T.mean(LP[T.arange(y.shape[0],y)]) is the mean (across minibatch examples) of the elements in v,
        #   i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given)[T.arange(y.shape[0]),y])
        
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                    'y should have the same shape as self.y_pred',
                    ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        
#%%
def load_data(dataset):
    """Load the dataset
    :param dataset: string, the path to the dataset(MNIST)
    """
    
    f = gzip.open(dataset,'rb')
    trainSet, validSet, testSet = pickle.load(f,encoding='bytes')
    f.close()
    
    def shared_dataset(dataXY, borrow=True):
        """Function that loads the dataset into theano variables
        
        The reason we store dataset in shared variable is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if thedata is not in in a shared vaiable) would
        lead to a large decrease in performance.
        """
        
        dataX, dataY = dataXY
        sharedX = theano.shared(numpy.asarray(dataX,dtype=theano.config.floatX),borrow=borrow)
        sharedY = theano.shared(numpy.asarray(dataY,dtype=theano.config.floatX),borrow=borrow)
        
        return sharedX, T.cast(sharedY, 'int32')
        
    trainSetX, trainSetY = shared_dataset(trainSet)
    validSetX, validSetY = shared_dataset(validSet)
    testSetX,  testSetY = shared_dataset(testSet)
    
    rval = [(trainSetX, trainSetY), (validSetX, validSetY), (testSetX, testSetY)]
    
    return rval
    
#%%
def SGD_Optimization_mnist(learning_rate=0.13, n_epocs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
   print('Loading data'+'.'*20)
   datasets =load_data(dataset)
   
   trainSet_x, trainSet_y = datasets[0]
   validSet_x, validSet_y = datasets[1]
   testSet_x,  testSet_y  = datasets[2]
   
   # compute number of minibatches for training, validation and testing
   n_train_batches = trainSet_x.get_value(borrow=True).shape[0] // batch_size
   n_valid_batches = validSet_x.get_value(borrow=True).shape[0] // batch_size
   n_test_batches  = testSet_x.get_value(borrow=True).shape[0] // batch_size
   
   print('Buliding the Model'+'.'*20)
   
   index = T.lscalar() # index to a [mini]batch
   
   # generate symbolic variables for input (x and y represent a minibatch)
   x = T.matrix('x')
   y = T.ivector('y')
   
   # Construct the logistic regerssion class
   # Each MNIST image has size 28*28, 10 classes totally
   datapoint_dim = 28*28
   classifier = LogisticRegression(input=x, n_in=datapoint_dim, n_out=10)
   
   # the cost we minimize during training is the negative log likelihood of the
   # model in symbolic format
   
   cost = classifier.negative_log_likelihood(y)
   
   test_model = theano.function(
       inputs=[index],
       outputs=classifier.errors(y),
       givens={
           x: testSet_x[index*batch_size:(index+1)*batch_size],
           y: testSet_y[index*batch_size:(index+1)*batch_size]
       }
   )
   
   valid_model = theano.function(
       inputs=[index],
       outputs=classifier.errors(y),
       givens={
           x: validSet_x[index*batch_size:(index+1)*batch_size],
           y: validSet_y[index*batch_size:(index+1)*batch_size]
       }
   )
   
   # compute the gradient of the cost with respect to theta = (W,b)
   g_W = T.grad(cost=cost,wrt=classifier.W)
   g_b = T.grad(cost=cost,wrt=classifier.b)
   
   # specify how to update the patameters of the model as a list of
   # (variable, update expression) pairs
   
   updates = [(classifier.W, classifier.W - learning_rate*g_W),
              (classifier.b, classifier.b - learning_rate*g_b)]
   
   train_model = theano.function(
       inputs=[index],
       outputs=cost,
       updates=updates,
       givens={
           x: trainSet_x[index*batch_size:(index+1)*batch_size],
           y: trainSet_y[index*batch_size:(index+1)*batch_size]
       }
   )
   
   print('Training the modal'+'.'*20)
   
   # early-stopping parameters
   patience = 8000
   patience_increase = 2
   improvement_threshold = 0.995
   validation_frequence = min(n_train_batches,patience / 2)
   
   best_validation_loss = numpy.inf
   test_score = 0.
   start_time = timeit.default_timer()
   
   done_looping = False
   epoch = 0
   while (epoch < n_epocs) and (not done_looping):
       epoch += 1
       for minibatch_index in range(n_train_batches):
           
           minibatch_avg_cost = train_model(minibatch_index)
           
           # iteration number
           iter = (epoch - 1) * n_train_batches + minibatch_index
           
           if (iter + 1) % validation_frequence == 0:
               validation_loss = [valid_model(i) for i in range(n_valid_batches)]
               this_validation_loss = numpy.mean(validation_loss)
               print(
                   'epoch %i, minibatch %i/%i, validation error %f %%' %
                   (epoch, minibatch_index+1, n_train_batches, this_validation_loss*100.))
                           
               if this_validation_loss < best_validation_loss:
                   if this_validation_loss < best_validation_loss * improvement_threshold:
                       patience = max(patience, iter * patience_increase)
                   best_validation_loss = this_validation_loss
                  
                  # test it on the test set
                  
                   test_losses = [test_model(i) for i in range(n_test_batches)]
                   test_score  = numpy.mean(test_losses)
                   print('       epoch %i, minibatch %i/%i, test error of best model %f %%' %
                    (epoch, minibatch_index+1, n_train_batches, test_score*100.))
                  
                   with open('best_model.pkl','wb') as f:
                       pickle.dump(classifier,f)
                      
               if patience <= iter:
                  done_looping = True
                  break
   end_time = timeit.default_timer()
   print(
      ('Optimization complete with best validation score of %f %%,'
      'with test performance %f %%'
      )
     %(best_validation_loss*100., test_score*100.)
   )
      
   print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))  
   print('The code for file ' + os.path.split(__file__)[1] + 
      ' ran for %.1fs' % (end_time - start_time) )
  
              
#%%
def predict():
    """
    An example of how to loda a trained model and use it to
    predict labels.
    """
    
    # load the save model
    classifier = pickle.load(open('best_model.pkl','rb'))
    # load a predictor function
    predictor_model = theano.function(
                            inputs=[classifier.input],
                            outputs=classifier.y_pred)
                            
    dataset = 'mnist.pkl.gz'
    datastes = load_data(dataset)
    tsetSetX, testSetY = datasets[2]
    testSetX = testSetX.get_value()
    
    predicted_values = predict_model(testSetX[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predict_values)
    
if __name__ == '__main__':
    SGD_Optimization_mnist()
                   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
