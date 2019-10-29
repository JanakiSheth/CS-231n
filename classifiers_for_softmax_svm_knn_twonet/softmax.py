import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in list(range(num_train)):
      scores = X[i].dot(W)  
      correct_score = scores[y[i]]
      loss += -np.log(np.exp(correct_score)/np.sum(np.exp(scores)))
  loss /= num_train
  loss += 0.5*reg*np.sum(W * W)
  
  scores_all = X.dot(W)
  for k in list(range(num_class)):
       partial_sum = np.exp(scores_all[:,k])/np.sum(np.exp(scores_all),axis = 1)    
       dW[:,k] = X.T.dot(partial_sum.reshape(scores_all.shape[0]))
  y_matrix = np.zeros((num_train,num_class))
  y_matrix[np.arange(num_train),y] = 1
  dW -=  (np.transpose(y_matrix).dot(X)).T

  dW /= num_train
  dW += reg* W  
        
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0] 
  num_class = W.shape[1]
  scores = X.dot(W)
  correct_score = scores[np.arange(num_train),y]#np.diagonal(np.take(scores, y, axis=1))  
  #import pdb; pdb.set_trace()
  loss += -np.sum(np.log(np.exp(correct_score)/np.sum(np.exp(scores), axis = 1)))
  loss /= num_train
  loss += 0.5*reg*np.sum(W * W)

  y_matrix = np.zeros((num_train,num_class))
  y_matrix[np.arange(num_train),y] = 1    
  denominator = np.sum(np.exp(scores),axis = 1, keepdims=True)    
  dW = X.T.dot(np.exp(scores)/denominator)#.reshape(scores.shape[0],1)) 
  dW -=  (np.transpose(y_matrix).dot(X)).T

  dW /= num_train
  dW += reg* W    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss,dW

