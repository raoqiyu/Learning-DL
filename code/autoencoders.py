# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 18:33:54 2015

Denoising Autoencoders
From Deep Learning Tutorial

The idea behind denoising autoencoder is simple. In order to force the hidden
layer to discover more robust features and prevent it from simply learning the
identity, we train the autoencoder to reconstruct the input from a corrupted
verion of it.
Two things:
    - Try to encode the input (preserve the information about the input)
    - Try to undo the effect of a corruption process stochastically applied
        to the input of the autoencoder

@author: wahah
"""

import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy

import os
import time
from logistic import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

class dA(object):
    """Denoising AutoEncoder class (dA)

    A denoising autoencoders tries to reconsruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space.

    x = c(input) # corrupted input
    y = s(Wx + b)
    z = s(W'y + b')
    """

    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784,
                 n_hidden=500, W=None, bhid=None, bvis=None):
        """
        Initialize the dA class by specifying:
            - the number of visible units(the dimension d of the input)
            - the number of hidden units(the dimension d' of the latent or hidden space)
            - the corruption level
        Symbolic variables for the input, weights and bias are useful when
        dealing with sdAs. The dA on layer 2 gets as input the output of the
        dA on layer 1.

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                shared belong the dA and another architecture; if dA should
                be shandalone set this to None.

        :type bhid: theano.tensor.Tensortype
        :param bhid: Theano variable pointing to a set of bias values (for
                hidden units) that should be shared belong the dA and another
                architecture; if dA should be shandalone set this to None.

        :type bvis: theano.tensor.Tensortype
        :param bvis: Theano variable pointing to a set of bias values (for
                visible units) that should be shared belong the dA and another
                architecture; if dA should be shandalone set this to None.

        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            W_value = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_value, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(n_visible,dtype=theano.config.floatX),
                borrow=True
            )
        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(n_hidden,dtype=theano.config.floatX),
                name='b', borrow=True
            )

        # y = s(Wx + b)
        # z = s(W'y + b')
        # z should be seen as a prediction (reconstruction) of x, given the
        # code y (hidden representation or distributed representation)
        # So the W' of the reverse mapping maybe constrained to be the
        # transpose of the forward mapping: self.W_prime = self.W.T
        self.W = W
        self.W_prime = self.W.T
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis

        self.theano_rng = theano_rng
        # if input is not given, generate a variable representing the input
        # Using the symbolic 'input' to the autoencoder as a parameter so that
        # we can concatenate layers of autoencoders to form a deep networks:
        # the symbolic output (the 'y' above) of layer 'k' will be the symbolic
        # input of layer 'k+1'
        if input is None:
            self.x = T.dmatrix('input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        """ Computes the value of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of hidden layer
        """
        return T.nnet.sigmoid(T.dot(hidden,self.W_prime) + self.b_prime)

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs
        of the same and zero-out randomly selected subset of size ``corruption_level``
        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_cost_updates(self, corruption_level, learning_rate):
        """This function computes the cost and the updates for one training
        step of the dA
        """

        tilde_x =self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        # Note: we sum over the size of a datapoint; if we are using minibatch,
        #       L will be a vector, with one entry per example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1-z), axis=1)
        # Note: L is now a vector, where each element is the cross-entropy
        #       cost of the reconstruction of the correnponding example of
        #       the minibatch. We need to get the cost of the minibatch.
        cost = T.mean(L)

        # compute the gradients of the cost of the 'dA' with respect
        # to its parameters
        gparams = T.grad(cost,self.params)
        updates = [(param, param - learning_rate * gparam)
                    for param, gparam in zip(self.params, gparams)
        ]
        return (cost, updates)

def testdA(learning_rate=0.1, training_epochs=15, dataset='mnist.pkl.gz',
           batch_size=20, output_folder='dA_plots'):

    print('Loading Data'+'.'*20)
    datasets = load_data(dataset)
    trainSetX, trainSetY = datasets[0]
#   validSetX, validSetY = datasets[1]
#   trainSetX, testSetY  = datasets[2]
    n_train_batches = trainSetX.get_value(borrow=True).shape[0] // batch_size

    index = T.lscalar()
    x = T.matrix('x')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    print('Building the Model With No Corruption '+'.'*20)
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28*28,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: trainSetX[index*batch_size:(index+1)*batch_size]
        }
    )

    start_time= time.clock()

    print('Training'+'.'*20)

    for epoch in range(n_train_batches):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
        print('Training epoch %d, cost '% epoch, numpy.mean(c))
    end_time = time.clock()
    training_time = (end_time - start_time)
    print('The no corruption code for file ' +
        os.path.split(__file__)[1] +
        ' ran for %.2fs' % training_time
    )
    img = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28,28),tile_shape=(10,10),
        tile_spacing=(1,1))
    )
    img.save('filter_corrupting_0.png')





    print('Building the Model With Corruption 30%'+'.'*20)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    rng = numpy.random.RandomState(123)

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28*28,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        inputs=[index],
        ouputs=cost,
        updates=updates,
        given={
            x: trainSetX[index*batch_size:(index+1)*batch_size]
        }
    )
    start_time= time.clock()

    print('Training'+'.'*20)

    for epoch in range(n_train_batches):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
        print('Training epoch %d, cost '% epoch, numpy.mean(c))
    end_time = time.clock()
    training_time = (end_time - start_time)
    print('The 30% corruption code for file ' +
        os.path.split(__file__)[1] +
        ' ran for %.2fs' % training_time
    )

    img = Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(28,28),tile_shape=(10,10),
        tile_spacing=(1,1))
    )
    img.save('filter_corrupting_30.png')
    os.chdir('../')

if __name__ == '__main__':
    testdA()