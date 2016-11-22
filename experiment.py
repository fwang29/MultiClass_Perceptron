import sys
import perceptron as P
import numpy as np
import matplotlib.pyplot as plt

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
    epochs = 20

    # arrays containing train and test
    train = np.array(parse_features('trainingimages', 5000))
    train_labels = np.array(parse_labels('traininglabels', 5000))
    test = np.array(parse_features('testimages', 1000))
    test_labels = np.array(parse_labels('testlabels', 1000))

    # initialize weight vectors
    w = np.array([[0 for i in range(784)]for j in range(10)])
    train_acc = [0 for i in range(epochs)]

    # training
    for j in range(epochs):
        mistakes = 0
        for i in range(5000):
            mistakes += P.perceptron(w, alpha, train[i], train_labels[i])
        
        train_acc[j] = (5000-mistakes)/(5000.0)      # accuracy so far
        print train_acc[j]
    # draw training accuracies in the end

    # testing
    accuracy = 0
    for i in range(1000):
        guess = P.perceptron_decision(w, test[i])
        if guess == test_labels[i]:
            accuracy += 1
    accuracy /= 1000.0
    print "test accuracy:"
    print accuracy

    # building confusion matrix
    confusion_counts = [[0 for i in range(10)] for j in range(10)]
    confusion_totals = [[0 for i in range(10)] for j in range(10)]
    confusion = [[0 for i in range(10)] for j in range(10)] 
    for i in range(0,1000):
        digit = test_labels[i]
        for j in range(0,10):
            confusion_totals[digit][j] += 1    # the whole digit row +=1
        guess = P.perceptron_decision(w, test[i])
        confusion_counts[digit][guess] += 1
    
    for i in range(0,10):
        for j in range(0,10):
            rate = (confusion_counts[i][j]+0.0) / confusion_totals[i][j]
            confusion[i][j] = "{0:.3f}".format(rate)
    for i in range(0,10):
        print confusion[i]

    # draw training accuracies
    plt.xlabel('num of epochs')
    plt.ylabel('training accuracy')

    l1 = plt.plot(range(epochs), train_acc)

    plt.title('Training Curve')
    plt.show()

