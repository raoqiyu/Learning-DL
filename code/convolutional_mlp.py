# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:30:32 2015

LeNet Convolutional Neural Network
From Deep Learning Tutorial

@author: wahah
"""

import theano
from theano import tensor as T
import numpy
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import os
import sys
import timeit


from logistic import LogisticRegression, load_data
from mlp import HiddenLayer

class LeNetConvPoolLayer(object):
    """Pool layer of a convolutional network """
    
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
        """
        Allocate a LeNetConvPoollayer with sshared variable parameters.
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, number of input feature maps,
                                     fiter height, filter width)
                                     
        :type image_shape: tuple or list of length  4
        :param image_shape: (batch_size, num input feature maps,
                                 image height, image width)
                                 
        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, @cols)        
        """
        
        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit        
        fan_in=numpy.prod(filter_shape[1:])
        # each unit in the lower layer receive a gradient from:
        # "num output feature maps * filter height * filter width / poolsize"        
        fan_out=(filter_shape[0] * numpy.prod(filter_shape[2:]) /
                    numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(rng.uniform(low=-W_bound,high=W_bound,
                size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        
        b_values = numpy.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filter, 1, 1). Each bias will
        # thus broadcasted across mini-batches and feature map width&height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))
        
        self.params = [self.W, self.b]
        
        self.input = input
  
def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz',
                    nkerns=[20,50],batch_size=500):
    """Demonstates lenet on MNIST dataset
    """
    
    rng = numpy.random.RandomState(1234)
    print('Loading Data'+'.'*20)
    datasets = load_data(dataset)
    
    trainSetX, trainSetY = datasets[0]
    validSetX, validSetY = datasets[1]
    testSetX,  testSetY  = datasets[2]
    
    n_train_batches = trainSetX.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = validSetX.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = testSetX.get_value(borrow=True).shape[0] // batch_size
    
    print('Building Data'+'.'*20)
    
    index = T.lscalar('index')
    x = T.matrix('x')
    y = T.ivector('y')
    
    # Reshape matrix of rasterized images of shape (batch_size, 28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size,1,28,28))
    
    # construct the first convolutional pooling layer
    # filtering reduces the image size to (28-5+1,28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(
        rng=rng,
        input=layer0_input,
        image_shape=(batch_size,1,28,28),
        filter_shape=(nkerns[0],1,5,5),
        poolsize=(2,2)
    )
    
    # construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size,nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(
        rng=rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],12,12),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2)
    )
    
    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        rng=rng,
        input=layer2_input,
        n_in=nkerns[1]*4*4,
        n_out=500,
        activation=T.tanh
    )
    
    layer3 = LogisticRegression(input=layer2.output,n_in=500,n_out=10)
    
    testModel = theano.function(
        inputs=[index],
        outputs=layer3.errors(y),
        givens={
            x:testSetX[index*batch_size:(index+1)*batch_size],        
            y:testSetY[index*batch_size:(index+1)*batch_size]        
        }    
    )
    validModel = theano.function(
        inputs=[index],
        outputs=layer3.errors(y),
        givens={
            x:validSetX[index*batch_size:(index+1)*batch_size],        
            y:validSetY[index*batch_size:(index+1)*batch_size]        
        }
    )
    
    params = layer3.params+layer2.params+layer1.params+layer0.params
    cost = layer3.negative_log_likelihood(y)
    grads = T.grad(cost,params)
    
    updates= [(param, param - learning_rate*grad)
                for param,grad in zip(params,grads)
    ]
    
    trainModel = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x:trainSetX[index*batch_size:(index+1)*batch_size],        
            y:trainSetY[index*batch_size:(index+1)*batch_size]        
        }    
    )
    
    print('Training'+'.'*20)
    
    patience = 10000
    patience_increase = 2
    improvement_threshold = 2
    validation_frequence = min(n_train_batches, patience/2)
    
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score =0.
    start_time = timeit.default_timer()
    
    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        
        for mini_batch_index in range(n_train_batches):
           iter = (epoch - 1) * n_train_batches + mini_batch_index
           if iter % 100 == 0:
               print('training @ iter = ' , iter)
           
           cost_ij = trainModel(mini_batch_index)
           if (iter + 1) % validation_frequence ==0:
               validation_losses = [validModel(i) for i 
                                       in range(n_valid_batches)]
               this_validation_losses = numpy.mean(validation_losses)
               print('epoch %i, minibatch %i/%i, validation error %f %%' %
                       (epoch, mini_batch_index+1, n_train_batches,
                        this_validation_losses*100)
               )
               if this_validation_losses < best_validation_loss:
                   best_validation_loss = this_validation_losses
                   best_iter = iter
                   if this_validation_losses < best_validation_loss * \
                                               improvement_threshold:
                       patience = max(patience, patience*patience_increase)
                   
                   test_losses = [testModel(i) for i in range(n_test_batches)]
                   test_score = numpy.mean(test_losses)
                   
                   print('   epoch %i, minibatch %i/%i, test error of'
                         'best model %f %%'%
                           (epoch, mini_batch_index+1, n_train_batches,
                            this_validation_losses*100)
                   )
           if patience <= iter:
               done_looping = True
               break
    endtime = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          ' with test performance %f %%' %
          (best_validation_loss*100, best_iter+1, test_score*100.)
    )
   
    print('The code for file ' + os.path.split(__file__)[1]+
          ' ran for %.2fm' % (endtime - start_time)/60.   
    )
   
if __name__ == '__main__':
    evaluate_lenet5()
                   
        
        
        
        
        
        
        
        
        
    