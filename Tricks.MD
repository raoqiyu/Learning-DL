#Tricks

1. Shared Variables
Theano supports transparent use of a GPU. it perform data-intensive calculations up to 140x faster when with CPU.(float32 only)  
So, if we want to use a GPU, we'd better store the data into GPU, since copying data into GPU is slow.  And if minibatch is used
, copying each minibatch everytime is need (if the data is not in a shared variable, that would lead to a large decrease in performance.  
```python
self.W = theano.shared(value=W_value,name='W', borrow=True)

2. **float32**
> Material below from Theano tutorial
"When using the GPU, **float32** tensor shared variables are stored on the GPU by default to elimate transfer time for GPU ops using those variables.  
The more **float32** variables are in your graph, the more work the GPU can do for you  
Consider adding floatX = float32 to your ~/.theanorc file if you plan to do a lot of work"  
```python
W_value = numpy.asarray(
    numpy_rng.uniform(...
    ),
    dtype=theano.config.floatX
)
