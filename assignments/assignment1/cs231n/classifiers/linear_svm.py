from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
from tqdm.autonotebook import tqdm

def svm_loss_naive(W, X, y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).
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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        loss_contributors_count = 0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                # incorrect class gradient part
                dW[:, j] += X[i]
                # count contributor terms to loss function
                loss_contributors_count += 1
        loss += np.sum(np.fmax(scores - correct_class_score + 1, 0)) - 1
        # correct class gradient part
        dW[:, y[i]] += (-1) * loss_contributors_count * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # Add regularization to the gradient
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  loss = 0.0
  dW = np.zeros(W.shape)  # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  # s: A numpy array of shape (N, C) containing scores
  margin = X.dot(W)
  # read correct scores into a column array of height N
  correct_score = margin[np.arange(num_train), y]
  correct_score = correct_score.reshape(num_train, -1)
  # subtract correct scores from score matrix and add margin
  # margin += np.ones(margin.shape) - np.outer(np.ones(num_classes), correct_score))
  margin += 1 - correct_score
  # make sure correct scores themselves don't contribute to loss function
  margin[np.arange(num_train), y] = 0
  margin = np.fmax(margin, 0)
  # construct loss function
  loss = np.sum(margin) / num_train
  loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  margin[margin > 0] = 1
  valid_margin_count = (margin).sum(axis=1)
  margin[np.arange(num_train),y] -= valid_margin_count
  dW = (X.T).dot(margin) / num_train +  reg * 2 * W
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  return loss, dW
