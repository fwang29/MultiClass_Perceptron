import sys
import perceptron as P
import numpy as np
#import matplotlib.pyplot as plt

def print_graph(test_id):
    start = test_id * 28
    end = start + 28
    file = open('testimages')
    for i, line in enumerate(file):
        if i >= start and i < end:
            print line
    file.close()

def parse_features(fname, N):
    arr = [[0 for k in xrange(28*28)]for i in xrange(N)]   # num cols, num rows, num examples
    file = open(fname)
    for i in range(0,N):
        for y in range(0,28):
            for x in range(0,29):    # need to consume the '\n' char
                char = file.read(1)
                if char == '+' or char == '#':
                    arr[i][y*28+x] = 1
    
    file.close()
    return arr

def parse_labels(fname, N):
    arr = []
    file = open(fname)
    for line in file:
        arr.append(int(line))
    file.close()
    return arr

if __name__ == '__main__':
    # params to tune
    bias = 0
    alpha = 1  # ? not sure learning rate decay function here
    epochs = 100

    # arrays containing train and test
    train = np.array(parse_features('trainingimages', 5000))
    train_labels = np.array(parse_labels('traininglabels', 5000))
    test = np.array(parse_features('testimages', 1000))
    test_labels = np.array(parse_labels('testlabels', 1000))

    # initialize weight vectors and mistakes
    weights = np.array([[0 for i in range(784)]for j in range(10)])

    # training
    for j in range(epochs):
        mistakes = 0
        for i in range(5000):
            mistakes += P.perceptron(weights, alpha, train[i], train_labels[i])
        
        print (5000-mistakes)/(5000.0)      # accuracy so far

    # testing
    accuracy = 0
    for i in range(1000):
        guess = P.perceptron_decision(weights, test[i])
        if guess == test_labels[i]:
            accuracy += 1
    accuracy /= 1000.0
    print accuracy
