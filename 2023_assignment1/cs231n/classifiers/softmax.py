from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    num_classes = W.shape[1]
    num_train = X.shape[0]



    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        denominator = 0.0
        max_value = max(scores)
        # 先把分母累加好，因为是先求指数，再累加，再求对数
        for j in range(num_classes):
            scores[j] -= max_value
            denominator += np.exp(scores[j])

        softmax_value = np.exp(scores[y[i]]) / denominator
        loss += - np.log(softmax_value)

        for j in range(num_classes):
            if j == y[i]:
                dW[:, y[i]] += X[i].T * (-1 + softmax_value)
            else:
                dW[:, j] +=  X[i].T * ( np.exp(scores[j]) / denominator)

    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)
    max_value = np.max(scores, axis=1).reshape((num_train,1)) # 找到每行的最大值
    scores -= max_value #减去每行最大值，防止除以大数导致数值不稳定
    scores_yi = scores[np.arange(num_train), list(y)].reshape(num_train, 1)

    loss_i = -scores_yi +  np.log(np.sum(np.exp(scores), axis=1).reshape((num_train, 1)))
    loss = np.sum(loss_i)
    loss = loss / num_train + reg * np.sum(W * W)

    margin = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape((num_train, 1))
    margin[np.arange(num_train), list(y)] -= 1;
    dW = X.T.dot(margin)
    dW = dW/num_train + 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
