# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 19:20:24 2015

Stacked Denoising Autoencoder (sdA)
From Deep Learning Tutorial

Denoising autoencoder can be stacked to form a deep network by feeding the
latent representation (output code) of the denoising autoencoder found on the
layer below as input to the current layer.

Stage 1 (Unsupervised pre-training):
    - Each layer is trained as a denoising autoencoder by minimizing the error
      in reconstructing its input (which is the output code of the previous
      layer). Once the first `k` layers are trained, we can train the `k+1`-th
      layer because we can now compute the code or latent representation from
      the layer below.

Stage 2 (Supervisied fine-tuning) :
    - In supervised fine-tuning we want to minimize prediction error on a
      supervisied task.
    - Firstly, we add a logistic regression layer on top of the network (more
      precisely on the output code of the output layer).
    - Then, we train the entire network as we would train a multilayer perceptron.

Two facades:
    - a list of autoencoders
    - an MLP
    In unsupervised pre-training stage, we use the first facade. In the second
    stage of training, we use the second facade. These two facades are linked because:
    - the autoencoders and the sigmoid layers of the MLP share parameters, and
    - the latent representations computed by intermediate layers of the MLP are
        fed as input to the autoencoders.

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


from logistic import LogisticRegression, load_data
from mlp import HiddenLayer
from autoencoders import dA

class SdA(object):
    """Stacked denoising autoencoder class (sdA)

    A stacked denoising autoencoder mode is obtained by stacking several dAs.
    The hidden layer of the dA at layer `i` becomes the input of the dA at layer
    `i+1`. The first layer dA gets as input the input of the sdA. and the hidden
    layer of the last dA represents the output. Note that after pretraining, the
    sdA is dealt with as a normal MLP, the dAs are only used to initialize
    the weights.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_in=784,
            hidden_layers_sizes=[500, 500], n_out=10,corruption_levels=[0.,0.1]):
        """This class is made to support a variable number of layers.

        :type  theano_rng: theano.tensor.shared_randomstream.RandomSteam
        :param theano_rng: Thenao random generator used to draw initial weights

        :type  n_in: int
        :param n_in: dimension of the input to the sdA

        :type  hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                                    at least one value

        :type  n_out: int
        :param n_out: dimension of the output of the network

        :type  corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each layer
        """
        # self.dA_layers will store the denoising autoencoder associated
        # with the layers of the MLP
        self.dA_layers = []
        # self.sigmoid_layers will store the sigmoid layers of the MLP facade
        self.sigmoid_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasteried images
        self.y = T.ivector('y') # the labels are presented as 1D vector of
                                # [int] labels

        #
        # Construct self.n_layers sigmoid layers and self.n_layers denoising
        # layersm where self.n_layers is the depth of our model
        #
        for i in range(self.n_layers):
            # construct a sigmoid layer
            #

            # the size of the input is ethier the number of the hidden units of
            # the layer below or the input size if we are on the first layer.
            # the input of the layer has the same situation
            if i == 0:
                input_size  = n_in
                layer_input = self.x
            else:
                input_size  = hidden_layers_sizes[i-1]
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layer
            self.sigmoid_layers.append(sigmoid_layer)
            # ??? the parameters of the sigmoid layers are paremeters of the
            # sdA, the visible bias in the dA are parameters of those
            # dA, but not the sdA. So we do not add the dA_layer's (below)
            # bvis to self.params.
            self.params.extend(sigmoid_layer.params)

            # construct a denoising autoencoder that shared weights with this
            # sigmoid_layer
            dA_layer = dA(numpy_rng=numpy_rng, theano_rng=theano_rng,
                          input=layer_input, n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W = sigmoid_layer.W, bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)

        #
        # Construct a logistic layer on top of the MLP
        #
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_out
        )
        self.params.extend(self.logLayer.params)

        #
        # Construct a function that impletements one step of finetuning
        #

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def preTraining_functions(self, trainSetX, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in training the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type  trainSetX: theano.tensor.TensorType
        :param trainSetX: Shared variable that contains all datapoints used
                            for training the dA

        :type  batch_size: int
        :param batch_size: size of a minibatch
        '''

        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('lr')
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(corruption_level,learning_rate)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.Param(corruption_level,default=0.2),
                        theano.Param(learning_rate,default=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: trainSetX[batch_begin:batch_end]
                }
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of finetuning,
        a function `validate` that computes the error on a batch from the validation
        set, and a function `test` that computes the error on a batch from the
        testing set

        :type  datasets: list of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                            that as to contain three pairs, `train`, `valid`,
                            `test` in this order, where each pair is formed of
                            two theano variables, one for the datapoints, the
                            other for labels

        :type  batch_size: int
        :param batch_size: learning_rate used during finetuns stage

        :type  learning_rate: float
        :param learning_rate: learning_rate used during finetune stage
        '''

        trainSetX, trainSetY = datasets[0]
        validSetX, validSetY = datasets[1]
        testSetX,  testSetY  = datasets[2]

        n_valid_batches = validSetX.get_value(borrow=True).shape[0] // batch_size
        n_test_batches  = testSetX.get_value(borrow=True).shape[0] // batch_size

        index = T.lscalar('index')
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        gparams = T.grad(self.finetune_cost, self.params)
        updates = [(param, param - learning_rate*gparam)
                    for param, gparam in zip(self.params, gparams)]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x :trainSetX[batch_begin:batch_end],
                self.Y :trainSetY[batch_begin:batch_end]
            },
            name='train'
        )

        valid_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x :validSetX[batch_begin:batch_end],
                self.Y :validSetY[batch_begin:batch_end]
            },
            name='valid'
        )

        test_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x :testSetX[batch_begin:batch_end],
                self.Y :testSetY[batch_begin:batch_end]
            },
            name='test'
        )

        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score

def test_SdA(finetune_lr=0.1, pretraining_epochs=15,
             pretraining_lr=0.001, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=1):

    print('Loading Data'+'.'*20)
    datasets =load_data(dataset)
    trainSetX, trainSetY = datasets[0]
    validSetX, validSetY = datasets[1]
    testSetX,  testSetY  = datasets[2]

    n_train_batches = trainSetX.get_value(borrow=True).shape[0] // batch_size

    print('Building Model'+'.'*20)
    numpy_rng = numpy.random.RandomState(1234)
    sda = SdA(
        numpy_rng=numpy_rng,
        n_in=28*28,
        hidden_layers_sizes=[1000,1000,1000],
        n_out=10
    )

    #
    #   Pretraining the Model
    #
    print('.'*10+'getting the pretraining functions'+'.'*10)
    pretraining_fns = sda.preTraining_functions(trainSetX=trainSetX,batch_size=batch_size)

    print('.'*10+'pretraining the model'+'.'*10)
    start_time = time.clock()
    corruption_levels = [.1, .2, .3]
    # pretraining layer-wise
    for i in range(sda.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretraining_lr))
            print('Pretraining layer %i, epoch %d, cost ' %(i, epoch), numpy.mean(c))
    end_time = time.clock()
    print('The pretraining code for file '+
            os.path.split(__file__)[1] +
          ' ran for %.2s' % (end_time - start_time))

    #
    # Finetuning the model
    #

    # get the training, validation, testing function for the model
    print('.'*10+'getting the finetune functions'+','*20)
    train_fn, valid_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('.'*20+'finetuning the model'+'.'*20)
    # early stop parameter
    patience = 10 * n_train_batches # look as this many examples regradless
    patience_increase = 2. # wait this much longer when a new best is found
    improvement_threshold = 0.95 # a relative improvement of this
                                 # much is consider significant
    validation_frequency = min(n_train_batches, patience/2)
                                    # go through this many minibatch
                                    # before checking the network
                                    # on the validation set
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while(epoch < training_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter - (epoch - 1) * n_train_batches + minibatch_index

            if (iter+1) % validation_frequency == 0:
                validation_losses = valid_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch, minibatch_index+1, n_train_batches,
                         this_validation_loss*100.)
                )

                if(this_validation_loss < best_validation_loss):
                    if(this_validation_loss < best_validation*improvement_threshold):
                        patience = max(patience, iter*patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_score = test_model()
                test_score = numpy.mean(test_score)
                print('    epoch %i, minibatch %i/%i, test error of '
                      'best model %f %%' %
                      (epoch, minibatch_index,n_train_batches,
                       test_score*100.))

            if patience <= iter:
                done_looping = True
                break
        end_time = time.clock()
        print('Optimization complete with best validation score of %f %%, '
              'on iteration %i, '
              'with test performance %f %%'%
              (best_validation_loss,best_iter+1,test_score*100.)
        )
        print('The training code for file ' +
                os.path.split(__file__)[1] +
                ' ran for #.2fs'%(end_time - start_time)
        )


if __name__ == '__main__':
    test_SdA()