import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_size=7,
               hidden_dim=100, num_aff = 2, num_con = 1, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_con = num_con
    self.num_layers = self.num_con+num_aff
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    C,H,W = input_dim
    self.params['W1'] = np.random.normal(0,weight_scale,(num_filters[0],C,filter_size,filter_size))
    self.params['b1'] = np.zeros((num_filters[0],))    
    for i in np.arange(1,num_con):
        self.params['W%d'%(i+1)] = np.random.normal(0,weight_scale,(num_filters[i],num_filters[i-1],filter_size,filter_size))
        self.params['b%d'%(i+1)] = np.zeros((num_filters[i-1],))
    i = self.num_con    
    self.params['W%d'%(i+1)] = np.random.normal(0,weight_scale,(num_filters[-1]*H*W // (4**(i)), hidden_dim))
    self.params['b%d'%(i+1)] = np.zeros((hidden_dim,))
    self.params['W%d'%(i+2)] = np.random.normal(0,weight_scale, (hidden_dim, num_classes))
    self.params['b%d'%(i+2)] = np.zeros((num_classes,))
    #import pdb; pdb.set_trace()
    if self.use_batchnorm:
        for i in np.arange(num_con):
             self.params['gamma%d' % (i+1)] = np.ones(num_filters[i])
             self.params['beta%d' % (i+1)] = np.zeros(num_filters[i])
        i = i+1    
        self.params['gamma%d' % (i+1)] = np.ones(hidden_dim)
        self.params['beta%d' % (i+1)] = np.zeros(hidden_dim)


    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_layers - 1)]
    
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    
    #W1, b1 = self.params['W1'], self.params['b1']
    #W2, b2 = self.params['W2'], self.params['b2']
    #W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = self.params['W1'].shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    cache_history = []
    
    mode = 'test' if y is None else 'train'    
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #

    if self.use_batchnorm:
        out_con_relu,cache_con_relu = conv_bn_relu_forward(X,self.params['W%d'%(1)],self.params['b%d'%(1)],self.params['gamma%d'%(1)],self.params['beta%d'% (1)],conv_param,self.bn_params[0])
    else:
        out_con_relu,cache_con_relu = conv_relu_forward(X,self.params['W%d'%(1)],self.params['b%d'%(1)],conv_param)
    cache_history.append(cache_con_relu)
    
    out_pool,cache_pool = max_pool_forward_fast(out_con_relu,pool_param)
    cache_history.append(cache_pool)
    
    for i in np.arange(1,self.num_con):
        if self.use_batchnorm:
            out_con_relu,cache_con_relu = conv_bn_relu_forward(out_pool,self.params['W%d'%(i+1)],self.params['b%d'%(i+1)],self.params['gamma%d'%(i+1)],self.params['beta%d'% (i+1)],conv_param,self.bn_params[i])
        else:
            out_con_relu,cache_con_relu = conv_relu_forward(out_pool,self.params['W%d'%(i+1)],self.params['b%d'%(i+1)],conv_param)
        cache_history.append(cache_con_relu)
        
        out_pool,cache_pool = max_pool_forward_fast(out_con_relu,pool_param)   
        cache_history.append(cache_pool)   
    """    
    if self.use_batchnorm:
        out_con_relu,cache_con_relu = conv_bn_relu_forward(out_pool,self.params['W%d'%(self.num_con)],self.params['b%d'%(self.num_con)],self.params['gamma%d'%(self.num_con)],self.params['beta%d'%(self.num_con)],conv_param,self.bn_params[self.num_con-1])
    else:
        out_con_relu,cache_con_relu = conv_relu_forward(out_pool,self.params['W%d'%(self.num_con)],self.params['b%d'%(self.num_con)],conv_param)
    cache_history.append(cache_con_relu)  
   """
    
    i = self.num_con
    #import pdb; pdb.set_trace()
    if self.use_batchnorm:
        out_aff_relu,cache_aff_relu = affine_bn_relu_forward(out_pool,self.params['W%d'%(i+1)],self.params['b%d'%(i+1)],self.params['gamma%d'%(i+1)],self.params['beta%d'% (i+1)],self.bn_params[i])
    else:
        out_aff_relu,cache_aff_relu = affine_relu_forward(out_pool,self.params['W%d'%(i+1)],self.params['b%d'%(i+1)])
    cache_history.append(cache_aff_relu)
        
    out_aff,cache_aff = affine_forward(out_aff_relu,self.params['W%d'%(i+2)],self.params['b%d'%(i+2)])  
    cache_history.append(cache_aff)
    
    scores = out_aff

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    loss, dscores = softmax_loss(scores,y)
    for j in np.arange(1,self.num_con+1):
        loss +=  0.5*self.reg*(np.sum(self.params['W%d'%j]*self.params['W%d'%j]))
        
    dx, grads['W%d'%(i+2)], grads['b%d'%(i+2)] = affine_backward(dscores, cache_history.pop())
    grads['W%d'%(i+2)] += self.reg*self.params['W%d'%(i+2)]
    
    if self.use_batchnorm:
        dx, grads['W%d'%(i+1)],grads['b%d'%(i+1)],grads['gamma%d'%(i+1)],grads['beta%d'%(i+1)] = affine_bn_relu_backward(dx, cache_history.pop())
    else:    
        dx, grads['W%d'%(i+1)],grads['b%d'%(i+1)] = affine_relu_backward(dx, cache_history.pop())
    grads['W%d'%(i+1)] += self.reg*self.params['W%d'%(i+1)]
    
    """
    if self.use_batchnorm:
        dx, grads['W%d'%(i)],grads['b%d'%(i)],grads['gamma%d'%(i)],grads['beta%d'%(i)] = conv_bn_relu_backward(dx, cache_history.pop())
    else:    
        dx, grads['W%d'%(i)],grads['b%d'%(i)] = conv_relu_backward(dx, cache_history.pop())
    grads['W%d'%(i)] += self.reg*self.params['W%d'%(i)]
    """
    for i in np.arange(self.num_con,0,-1):
        dx = max_pool_backward_fast(dx, cache_history.pop())
        if self.use_batchnorm:
            dx, grads['W%d'%(i)],grads['b%d'%(i)],grads['gamma%d'%(i)],grads['beta%d'%(i)] = conv_bn_relu_backward(dx, cache_history.pop())
        else:    
            dx, grads['W%d'%(i)],grads['b%d'%(i)] = conv_relu_backward(dx, cache_history.pop())
        grads['W%d'%(i)] += self.reg*self.params['W%d'%(i)]
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
