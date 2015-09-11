# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 20:56:31 2015

Multilayer Perceptron
From Deep Learning Tutorial

@author: wahah
"""

import theano
import theano.tensor as T
import numpy

import time

from logistic import *

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out) 
        and the bias vector b is of shape (n_out,).
        """
        
        self.input = input
        
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size = (n_in,n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values,name='W',borrow=True)
        if b is None:
            b_values = numpy.zeros((n_out,),dtype=theano.config.floatX)
            b = theano.shared(value=b_values,name='b',borrow=True)
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        
        linear_out = T.dot(input,self.W) + self.b
        self.output = (
            linear_out if activation is None
            else activation(linear_out)
        )
        
class MLP(object):
    """Multi-Layer Perceptron Class
    
    A multilayer perceptron is a feedforward artifical neural model that has
    one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have a activation function tanh or the 
    sigmoid function (defined here by a ''HiddenLayer'' class) while the 
    top layer is a softman layer (defined here by a ''LogisticRegression'' 
    class)        
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        
        self.logRegressionLayer= LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        
        self.L1 = (
            abs(self.hiddenLayer.W)
            + abs(self.logRegressionLayer.W)
        )
        
        self.L2_sqr = (
            (self.hiddenLayer.W**2).sum()
            + (self.logRegressionLayer.W**2).sum()
        )
        
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        
        self.errors = self.logRegressionLayer.errors
        
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        
def Test_MLP(learning_rate = 0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden = 500):

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
   rng = numpy.random.RandomState(1234)
   classifier = MLP(rng=rng,input=x, n_in=datapoint_dim,n_hidden=n_hidden, n_out=10)
   
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
   gparams = [T.grad(cost,param) for param in classifier.params]
   
   # specify how to update the patameters of the model as a list of
   # (variable, update expression) pairs
   
   updates = [(param, param - learning_rate*gparam)
               for param, gparam in zip(classifier.params,gparams)]
   
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
   patience = 10000
   patience_increase = 2
   improvement_threshold = 0.995
   validation_frequence = min(n_train_batches,patience / 2)
   
   best_validation_loss = numpy.inf
   test_score = 0.
   start_time = timeit.default_timer()
   
   done_looping = False
   epoch = 0
   while (epoch < n_epochs) and (not done_looping):
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
                 
if __name__ == '__main__':
    Test_MLP()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        