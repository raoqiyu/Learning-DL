# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 20:34:21 2015

Recurrent Neural Network
From Deep Learning Tutorial

Elman recurrent neural network
It takes as input the recurrent input (time t) and the previous hiddent state
(time t-1). Then it iterates.

So, we should process the input to fit this sequential/temporal structure. It
consisit on a matrix where the row 0 corresponds to the time step t=0, the row
1 corresponds to the time step t=1.

The parameter of the E-RNN to be learned are:
    - the word embedding (real-valued matrix) : x(t) --> h(t)
    - the initial hidden state (real-value vector)
    - two matrices for the linear projection of the input t and the previous
        hidden layer state t-1 :  h(t-1) ---> x(t)
    - (optional) bias. Recommendation: don't use it.
    - softmax classification layer on top

The hyperparameters define the whole architecture:
    - dimension of the word embedding
    - size of the vocabulary
    - number of hidden units
    - number of classes
    - random seed + way to initialize the model

@author: wahah
"""

import os
import sys
import random
import gzip
import pickle
import time
import subprocess
import copy
from functools import reduce
import numpy
import theano
from theano import tensor as T
from collections import OrderedDict

sys.setrecursionlimit(1500)
##################################

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)


def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out
# end-snippet-1

def get_perf(filename, folder):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = os.path.join(folder, 'conlleval.pl')
  #  print(_conlleval)
    if not os.path.isfile(_conlleval):
        url = 'http://www-etud.iro.umontreal.ca/~mesnilgr/atis/conlleval.pl'
        download(url, _conlleval)
        os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                            _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(b''.join(open(filename,'rb').readlines()))
    for line in stdout.split(b'\n'):
        if b'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}

# metrics function using conlleval.pl
def conlleval(p, g, w, filename, script_path):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score

    OTHER:
    script_path :: path to the directory containing the
    conlleval.pl script
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            w = w.decode() ; wl = wl.decode(); wp = wp.decode()
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename, script_path)

# data loading functions
def atisfold():
    filename = 'atis.fold3.pkl.gz'
    f = gzip.open(filename, 'r')
    train_set, valid_set, test_set, dicts = pickle.load(f,encoding='bytes')
    return [train_set, valid_set, test_set, dicts]


##################################

class RNNSLU(object):
    ''' elman neural net model '''
    def __init__(self, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size
        '''

        #parameters of the model

        #
        # h[t] = sigmoid( wx*x[t] + wh*h[t-1] + bh )
        # h[0] is  initialized with [0,0,...,0]
        # output[t] = softmax( w*h[t] + b ) # a multi-class task
        # x[t] = emb[input[t]]
        # emb : each word has a embedding representation
        #
        #

        self.emb = theano.shared(name='embeddings',
                                 value=0.2*numpy.random.uniform(-1.0, 1.0,
                                 (ne+1, de)).astype(theano.config.floatX))

        self.wx = theano.shared(name='wx', value=0.2*numpy.random.uniform(
                                -1.0, 1.0, (de * cs, nh)).astype(theano.config.floatX))

        self.wh = theano.shared(name='wh', value=0.2*numpy.random.uniform(
                                -1.0, 1.0, (nh, nh)).astype(theano.config.floatX))

        self.w  = theano.shared(name='w', value=0.2*numpy.random.uniform(
                                -1.0, 1.0, (nh, nc)).astype(theano.config.floatX))

        self.bh  = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        self.b  = theano.shared(name='b',
                                value=numpy.zeros(nc,
                                dtype=theano.config.floatX))

        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        self.params = [self.emb, self.wx, self.wh, self.w, self.bh, self.b, self.h0]

        ##
        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0],de*cs))
        y_sentence = T.ivector('y_sentence') # labels
        ##

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t,self.wx) + T.dot(h_tm1, self.wh)
                                    + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])
        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)


        ##
        # cost and gradient
        lr = T.scalar('lr')
        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                                [T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr *g)
                                        for p, g in zip(self.params, sentence_gradients))

        ##
        # theano function to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)

        ##
        #
        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                             self.emb /
                                             T.sqrt((self.emb**2).sum(axis=1))
                                             .dimshuffle(0,'x')})

        ##
    def train(self, x, y, window_size, learning_rate):
        cwords = contextwin(x, window_size)
        words = list(map(lambda x : numpy.asarray(x,dtype='int32'), cwords))
        labels = y
        self.sentence_train(words, labels, learning_rate)
        self.normalize()

    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder, param.name + '.npy'),
                           param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                                            param.name + '.npy')))


def testRNNSLU(param=None):
    if not param:
        param = {
            'fold' : 3,
            'data' : 'atis',
            'verbose' : 1,
            'lr' : 0.097,
            'decay' : True,
            'win' : 7,
            'nhidden' : 300,
            'seed' : 345,
            'emb_dimension' : 50,
            'nepoch' : 80,
            'savemodel' : False
        }
    print(param)

    folder_name = os.path.basename(__file__).split('.')[0]
 #   print(folder_name)
    folder = os.path.join(os.path.dirname(__file__), folder_name)
 #   print(folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    # load the  dataset
    trainSet, validSet, testSet, dic = atisfold()

    idx2label = dict((k, v) for v, k in dic[b'labels2idx'].items())
    idx2word  = dict((k, v) for v, k in dic[b'words2idx'].items())

    train_lex, train_ne, train_y = trainSet
    valid_lex, valid_ne, valid_y = validSet
    test_lex,  test_ne,  test_y  = testSet

    vocSize = len(set(reduce(lambda x, y : list(x) + list(y),
                              train_lex + valid_lex + test_lex)))
    nclasses = len(set(reduce(lambda x, y : list(x) + list(y),
                              train_y + valid_y + test_y)))
    nsentence = len(train_lex)

    groundtruth_valid = [list(map(lambda x: idx2label[x], y)) for y in valid_y]
    words_valid = [list(map(lambda x: idx2word[x], w)) for w in valid_lex]
    groundtruth_test = [list(map(lambda x: idx2label[x], y)) for y in test_y]
    words_test = [list(map(lambda x: idx2word[x], w)) for w in test_lex]


    ##
    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    rnn = RNNSLU(nh=param['nhidden'],
                 nc=nclasses,
                 ne=vocSize,
                 de=param['emb_dimension'],
                 cs=param['win']
    )

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    param['clr'] = param['lr']
    for epoch in range(param['nepoch']):
        # shuffle
        shuffle([train_lex, train_ne, train_y], param['seed'])
        sys.stdout.flush()
        param['ce'] = epoch
        tic = time.time()

        for i, (x, y) in enumerate(zip(train_lex, train_y)):
            rnn.train(x, y, param['win'], param['clr'])
        print('[learning] epoch %i >> 100%%' % epoch,
              ' completed in %.2f (sec) << \r' % (time.time() - tic))
        sys.stdout.flush()

        predictions_test = [list(map(lambda x: idx2label[x],
                               rnn.classify(numpy.asarray(
                               contextwin(x,param['win'])).astype('int32'))))
                           for x in test_lex]

        predictions_valid = [list(map(lambda x: idx2label[x],
                               rnn.classify(numpy.asarray(
                               contextwin(x,param['win'])).astype('int32'))))
                           for x in valid_lex]

       # evaluation // compute the accuracy using conlleval.pl
        res_test = conlleval(predictions_test,
                             groundtruth_test,
                             words_test,
                             folder + '/current.test.txt',
                             folder)
        res_valid = conlleval(predictions_valid,
                              groundtruth_valid,
                              words_valid,
                              folder + '/current.valid.txt',
                              folder)

        if res_valid['f1'] > best_f1:

            if param['savemodel']:
                rnn.save(folder)

            best_rnn = copy.deepcopy(rnn)
            best_f1 = res_valid['f1']

            if param['verbose']:
                print('NEW BEST: epoch', epoch,
                      'valid F1', res_valid['f1'],
                      'best test F1', res_test['f1'])

            param['vf1'], param['tf1'] = res_valid['f1'], res_test['f1']
            param['vp'], param['tp'] = res_valid['p'], res_test['p']
            param['vr'], param['tr'] = res_valid['r'], res_test['r']
            param['be'] = epoch

            subprocess.call(['mv', folder + '/current.test.txt',
                            folder + '/best.test.txt'])
            subprocess.call(['mv', folder + '/current.valid.txt',
                            folder + '/best.valid.txt'])
        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break

    print('BEST RESULT: epoch', param['be'],
          'valid F1', param['vf1'],
          'best test F1', param['tf1'],
          'with the model', folder)
    

if __name__ == '__main__':
    testRNNSLU()
