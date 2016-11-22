import numpy as np
import math

def perceptron_decision(w, x):    # doesn't matter adding bias or not?
    """returns c = argmaxc dot(wc, x)
    @param w: the weights vector
    @param x: the instance
    """
    cmax = 0
    c = 0    
    for i in range(10):
        tmp = np.dot(np.transpose(w[i]), x)
        if tmp > cmax:
            cmax = tmp
            c = i
    return c


def perceptron(w, r, x, y):
    """updates w by perceptron when needs update 
    @param w: the weights vector
    @param r: the learning rate
    @param x: the instance
    @param y: the label
    """
    c = perceptron_decision(w, x)
    if y != c:    # misclassification -> needs update, else do nothing
        w[c] -= r*x
        for i in range(10):
            if i != c:
                w[i] += r*x
        return 1
    return 0
