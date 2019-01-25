import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
         that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    
    python range()와 xrange() 차이
    - https://bluese05.tistory.com/57
    """
    
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)       # (D,) * (D,C) = (C,)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,y[i]] -= X[i,:].T     # Loss 만드는 W 약화
                dW[:,j] += X[i,:].T        # 정답 만드는 W 강화
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train
    loss /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    
    # Divide the gradient by the number of training examples
    dW /= num_train
    
    # Add the gradient of regularization
    dW += reg*W    
    
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
    """
    Structured SVM loss function, vectorized implementation.
    
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = X.dot(W)        # (N,D) * (D,C) = (N, C)
    
    num_train = X.shape[0]
    correct_class_score = scores[np.arange(num_train),y].reshape(-1,1)    # (N,1)
    margins = scores - correct_class_score + 1      # (N,C)
    margins = np.maximum(0,margins)
    margins[np.arange(num_train),y] = 0
    
    # short version
    # margins = np.maximum(0, X.dot(W)-X.dot(W)[np.arange(num_train),y].reshape(-1,1)+1)
    # margins[np.arange(num_train),y] = 0
    
    
    loss = np.sum(margins)
    
    # Compute the average
    loss /= num_train
    
    # Add regularization
    loss += reg * np.sum(W*W)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    margins[margins>0] = 1    # score는 필요없고 0보다 큰 margin에만 gradient 적용 위함
    margins[np.arange(num_train),y] = -np.sum(margins, axis=1)   # 위에서 1로 다 바꿨으니
    # sum을 하면 정답 label을 제외한 나머지 개수가 됨, -인 이유는 정답을 만들어주는 gradient를
    # 더 강화해주기 위함
    dW=X.T.dot(margins)         # (D, N) * (N,C) = (D,C) 이 과정에서 margin은 자동으로 덧셈이 된다.
    # Divide by the number of training examples
    dW /= num_train            
    # Add regularization
    dW += reg*W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return loss, dW
