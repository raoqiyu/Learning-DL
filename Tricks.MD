#Tricks

___Shared Variables___  
Theano supports transparent use of a GPU. it perform data-intensive calculations up to 140x faster when with CPU.(_float32_ only)  
So, if we want to use a GPU, we'd better store the data into GPU, since copying data into GPU is slow.  And if minibatch is used, copying each minibatch everytime is need (if the data is not in a shared variable, that would lead to a large decrease in performance.
```python
self.W = theano.shared(value=W_value,name='W', borrow=True)
```  

___float32___
>Material below from Theano tutorial  
"When using the GPU, _float32_ tensor shared variables are stored on the GPU by default to elimate transfer time for GPU ops using those variables.  
The more _float32_ variables are in your graph, the more work the GPU can do for you.  
Consider adding _floatX = float32_ to your _~/.theanorc_ file if you plan to do a lot of work"  

```python
W_value = numpy.asarray(
    numpy_rng.uniform(...
    ),
    dtype=theano.config.floatX
)
```
