# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:08:18 2015

Long Short Term Memory
From Deep Learning Tutorial

In a traditional recurrent neural network, during the gradient signal can end up
being multiplied a large number of times (as the number of timesteps) by the
weight matrix associated with the connections between the neurons of the recurrent
hidden layer. This means that, the magnitude of weights in the transtion can
have a strong impact on the learning process.

Vanishing Gradient ---- the gradient signal gets so small that learning either
becomes very slow or stops working altoghter. It can also make more difficult
the task of learning long-term dependencies in the data.

Exploding Gradients --- the weights in the matrix are large, it can lead to a
situation where the gradient signal is so large that it can cause learning to
diverge.

Memory Cells:
    - A neuron with a self-recurrent connection: a connection to itself,
        has a weight of 1.0 (can remain constant from one timestep from another.
                    )
    - Input  Gate: allow incoming signal to alter the state of the memory cell
         or block it
    - Output Gate: allow the state of the memory cell to have an effect on other
        neuron or prevent it.
    - Forget Gate: modulate the memory cell's self-recurrent connection, allowing
        the cell to remember or forget its previous state, as needed.

Equations:
x(t) is the input to the memory cell layer at time t
Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo and Vo  are weight matrices
bi, bf, bc and bo are bias vectors

    i(t)  = sigmoid( Wi*x(t) + Ui*h(t-1) + bi)
    C'(t) = tanh( Wc*x(t) + Uc*h(t-1) + bc)
    f(t)  = sigmoid( Wf*x(t) + Uf*h(t-1) + bf)

    C(t)  = i(t) * C'(t) + ft * C(t-1)
    o(t)  = sigmoid(Wo*x(t) + Uo*h(t-1) + Vo*C(t) + bo)
    # o(t)  = sigmoid(Wo*x(t) + Uo*h(t-1) + bo)
    h(t)  = ot * tanh(C(t))


Implementation Note:
We can concatentate the four matrices W* into a single weight matrix W and
performing the same concatenation on the weight matrices U* to produce the matrix
U and the bias vectors b* to produce the vector b, Then
    z = sigmoid( W*x(t) + U*h(t-1) + b).
The result is then sliced to obtain the pre-nonlinearity activations for i(t),
f(t), C'(t) and o(t).


@author: wahah
"""

import time
from collections import OrderedDict
import pickle
import numpy
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb
SEED = 1234
numpy.random.seed(SEED)

datasets = {'imdb':(imdb.load_data, imdb.prepare_data)}

#%%  utils function

# data process
def np_floatXArr(data):
    return numpy.asarray(data, dtype=theano.config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return list(zip(range(len(minibatches)), minibatches))


def get_dataset(name):
    return datasets[name][0], datasets[name][1]

# params
def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def dropout_layer(state_before, use_noise, trng):
    proj = T.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(theano.config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(theano.config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(theano.config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)
# lstm param


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(theano.config.floatX)

    return params
#%% Lstm
def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = T.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = T.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = T.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (T.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[T.alloc(np_floatXArr(0.),
                                                           n_samples,
                                                           dim_proj),
                                              T.alloc(np_floatXArr(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


layers = {'lstm':(param_init_lstm, lstm_layer)}
#%% Feed Forward (normal neural net), only useful to put after lstm before the classifier

def sgd(lr, tparams, grads, x, mask, y, cost):
    '''Stochasic Gradient Descent
    '''

    # Net set of shared variable that will contain the gradient
    # for a minibatch
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
                for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared= theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')
    pup = [(p, p-lr*g) for p, g in zip(list(tparams.values()), gshared)]

    # Function that updates the weights from the previously computed
    # gradient
    f_update = theano.function([lr], updates=pup, name='sgd_f_updates')

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * np_floatXArr(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * np_floatXArr(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * np_floatXArr(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

#%% Build Model

def build_model(tparams, options):
    rng = RandomStreams(SEED)

    # Use for dropout.
    use_noise = theano.shared(np_floatXArr(0.))

    x = T.matrix('x', dtype='int64')
    mask = T.matrix('mask', dtype=theano.config.floatX)
    y = T.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                                prefix=options['encoder'],
                                                mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, rng)

    pred = T.nnet.softmax(T.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = - T.log(pred[T.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_errors(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - np_floatXArr(valid_err) / len(data[0])

    return valid_err

def train_lstm(
    dim_proj=128, # word embeding dimension and LSTM number of hidden units.
    patience=10,   # Number of epoch to wait before early stop if no progress
    max_epochs=5000, # The maximum number of epoch to run
    dispFreq=10,    # display to stdout the training progress every N updates
    decay_c=0.,    # Weight decay for the classifier applied to the U weights
    lrate=0.0001,  # Learning rate for sgd
    n_words=10000, # Vocabulary size
    optimizer=sgd,
    encoder='lstm',
    saveto='lstm.npz', # the best mode will be saved there
    validFreq=370, # Compute the validation error after this number of update.
    saveFreq=1110, # Save the parameters after every saveFreq updates
    maxlen=100,  # sequence longer than this get ignored
    batch_size=16, # The batch size during training
    valid_batch_size=64, # The batch size used for validation/test set.
    dataset='imdb',

    #Parameter for extra option
    noise_std = 0.,
    use_dropout = True, # if False slightly faster, but worst test error
                            # This frequently need a bigger model.
    reload_model = None, # Path to a saved model we want to start from
    test_size=-1 # if > 0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    load_data, prepare_data = get_dataset(dataset)

    print('Loading data')
    train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
                                   maxlen=maxlen)
    if test_size > 0:
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    ydim = numpy.max(train[1]) + 1

    model_options['ydim'] = ydim

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano shared variable from the parameters
    tparams = init_tparams(params)

    (use_noise, x, mask, y , f_pred_prob, f_pred, cost) = build_model(tparams,model_options)

    if decay_c > 0.:
        decay_c = theano.shared(np_floatXArr(decay_c),name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U']**2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = T.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = T.scalar('lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, mask, y, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]),valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]),valid_batch_size)

    print('%d train examples'%len(train[0]))
    print('%d valid examples'%len(valid[0]))
    print('%d test examples'%len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    uidx = 0
    early_stop = False
    start_time = time.clock()

    try:
        for epoch in range(max_epochs):
            n_samples = 0

            # Get new shuffle index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random example for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                x, mask, y = prepare_data(x,y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x,mask,y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('NaN detected')
                    return 1., 1., 1.

                if numpy.mod(uidx,dispFreq) == 0:
                    print('Epoch ', epoch, 'Update ', uidx, 'Cost ', cost)

                if saveto and numpy.mod(uidx,saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl'%saveto,'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_errors(f_pred, prepare_data, train, kf)
                    valid_err = pred_errors(f_pred, prepare_data, valid,
                                           kf_valid)
                    test_err = pred_errors(f_pred, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (uidx == 0 or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err)

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            early_stop = True
                            break

            print('Seen %d samples' % n_samples)

            if early_stop:
                break

    except KeyboardInterrupt:
        print( "Training interupted")

    end_time = time.clock()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_errors(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_errors(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_errors(f_pred, prepare_data, test, kf_test)

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (epoch + 1), (end_time - start_time) / (1. * (epoch + 1))))
    print('Training took %.1fs' %(end_time - start_time))
    return train_err, valid_err, test_err

if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=100,
        test_size=500,
	optimizer=adadelta,
    )

