# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 18:49:29 2015

@author: wahah
"""

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy

import time
import sys

theano.config.floatX = 'float64'
#%%
# The Convolution Operator

rng = numpy.random.RandomState(4567)

# instantiate 4D tensor for input 
# (mini_batch_size, 3 feature maps(RGB), image height(900), image width(1400))
input = T.tensor4(name='input')

# initialize shared variable for weight
# (2 feature maps in hidden layer, 3(RGB) feature maps, filter size, filter size)
w_shp = (2, 3, 9, 9)
w_bound = numpy.sqrt(3*9*9)

W = theano.shared(numpy.asarray(
            rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_shp),
            dtype=input.dtype), name = 'W')
            
# initialize shared variable for bias (1D tensor) with random values
# import: bias are usually initialized to zero.
b_shp = (2,)
b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input.dtype), name='b')
conv_out = conv.conv2d(input,W)
output = T.nnet.sigmoid(conv_out+b.dimshuffle('x',0,'x','x'))

f = theano.function([input],output)            
ff = theano.function([input],conv_out)            
            
#%%

import pylab
from PIL import Image

img = Image.open('1.jpg')
img = numpy.asarray(img,dtype='float64') / 256.

img_ = img.transpose(2,0,1).reshape(1,3,900,1440)
filtered_img = f(img_)
filtered_img2 = ff(img_)

pylab.subplot(1,3,1);pylab.axis('off');pylab.imshow(img)
pylab.gray();
pylab.subplot(1,3,2);pylab.axis('off');pylab.imshow(filtered_img[0,0,:,:])
pylab.imsave('2.jpg',filtered_img[0,0,:,:])
pylab.subplot(1,3,3);pylab.axis('off');pylab.imshow(filtered_img[0,1,:,:])
pylab.imsave('3.jpg',filtered_img[0,1,:,:])
pylab.show()
