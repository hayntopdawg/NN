from __future__ import division
import csv
import numpy as np
import sys

__author__ = 'Jamie Fujimoto'
# Used code from:
# http://iamtrask.github.io/2015/07/12/basic-python-network/
# http://arctrix.com/nas/python/bpnn.py


def read_data(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield row


# Splits data into feature matrix and class vector
def split_data(filename):
    X, y = [], []
    for i, row in enumerate(read_data(filename)):
        X.append(row[:-1])
        y.append(row[-1])
    return np.array(X, dtype='float'), np.array(y)


# Used for output layer (no)
def get_num_classes(y):
    return len(set(y))


# Removes duplicates and keeps classes in same order
def class_list(y):
    known = set()
    newlist = []

    for c in y:
        if c in known: continue
        newlist.append(c)
        known.add(c)

    return newlist


# Converts y into a binary matrix
def y_to_binary(y):
    c_matrix = np.zeros([y.shape[0], get_num_classes(y)])
    classes = class_list(y)
    for i, c in enumerate(y):
        c_matrix[i, classes.index(c)] = 1
    return c_matrix


def get_class(pred):
    c = np.zeros(pred.shape)
    if pred.shape[1] > 1:
        c[0, np.argmax(pred)] = 1
    else:
        # prediction is closer to 1 than 0
        if 1 - pred < pred:
            c[0] = 1
    return c


# Gets accuracy given predictions and true values
def accuracy(preds, y):
    # print preds.shape
    correct = 0
    for i, p in enumerate(preds):
        # print p.shape
        c = get_class(p)
        if c.all() == y[i].all():
            correct += 1
    return correct / preds.shape[0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NN():
    # Assume only one hidden layer of nh neurons
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output neurons
        self.ni = ni + 1  # + 1 for the bias node
        self.nh = nh + 1  # + 1 for the bias node
        self.no = no

        # initialize activations
        self.ai = np.ones((self.ni, 1))
        self.ah = np.ones((self.nh, 1))
        self.ao = np.ones((self.no, 1))

        # seed random numbers to make calculation deterministic
        # np.random.seed(1)

        # initialize weights with random values between -0.5 and 0.5
        self.wi = np.random.rand(self.ni, self.nh - 1) - 0.5  # ni x (nh - 1) matrix
        self.wh = np.random.rand(self.nh, self.no) - 0.5  # nh x no matrix


    # Input Xi vector, feed forward through NN, and get a prediction
    def feedforward(self, x):
        # Set input neurons
        for i in xrange(self.ni - 1):
            self.ai[i] = x[i]

        # Calculate hidden neurons
        for j in xrange(self.nh - 1):
            s = 0
            for i in xrange(self.ni):
                s += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(s)

        # Calculate output neurons
        for j in xrange(self.no):
            s = 0
            for i in xrange(self.nh):
                s += self.ah[i] * self.wh[i][j]
            self.ao[j] = sigmoid(s)

        return self.ao.copy()


    def backpropagate(self, t, eta):
        # Case 1
        # calculate output deltas
        do = np.ones(self.no)
        output_deltas = np.ones((self.nh, self.no))
        for j in xrange(self.no):
            do[j] = (t[j] - self.ao[j]) * self.ao[j] * (1 - self.ao[j])
            for i in xrange(self.nh):
                output_deltas[i][j] = eta * do[j] * self.ah[i]

        # update hidden->output weights
        for i in range(self.nh):
            for j in range(self.no):
                self.wh[i][j] += output_deltas[i][j]

        # Case 2
        # calculate hidden deltas
        dh = np.ones(self.nh - 1)
        hidden_deltas = np.ones((self.ni, self.nh - 1))
        for j in xrange(self.nh - 1):
            s = 0
            for k in xrange(self.no):
                s += do[k] * self.wh[j][k]
            dh[j] = self.ah[j] * (1 - self.ah[j]) * s
            for i in xrange(self.ni):
                hidden_deltas[i][j] = eta * dh[j] * self.ai[i]

        # update input->hidden weights
        for i in range(self.ni):
            for j in range(self.nh - 1):
                self.wi[i][j] += hidden_deltas[i][j]


    def train(self, X, y, eps, eta, epochs):
        """
        :param eps: epsilon value for convergence of the backprop inner loop
        :param eta: positive constant (learning rate)
        :param epochs: number of runs through the entire dataset
        """
        for e in xrange(epochs - 1):
            r = np.arange(X.shape[0])
            np.random.shuffle(r)
            for i in r:
                converged = False
                itr = 0
                while not converged:
                    itr += 1
                    y_hat = self.feedforward(X[i])
                    E = 0.5 * (np.linalg.norm(y_hat.T - y[i]) ** 2)
                    if E <= eps:
                        converged = True
                    else:
                        if itr % 1000 == 0:
                            print E
                        self.backpropagate(y[i], eta)


    def predict(self, X):
        return self.feedforward(X).T


def iris():
    # Input (X) and target (y) datasets
    X, y = split_data('iris.data.txt')

    # Convert y into a binary matrix
    c = y_to_binary(y)

    # Create NN
    n = NN(X.shape[1], X.shape[1], c.shape[1])

    # Train NN
    n.train(X, c, eps=0.001, eta=0.1, epochs=100)

    # Test NN
    preds = []
    for i in xrange(X.shape[0]):
        pred = n.predict(X[i])
        preds.append(pred)
        print "{0} -> {1} ~ {2}:{3}".format(X[i], pred, c[i], y[i])

    print "Accuracy: {0}".format(accuracy(np.array(preds), c))


def iris_virginica():
    # Input (X) and target (y) datasets
    X, y = split_data('iris-virginica.txt')

    # Convert y into a binary matrix
    c = y_to_binary(y)

    # Create NN
    n = NN(X.shape[1], X.shape[1], c.shape[1])

    # Train NN
    n.train(X, c, eps=0.01, eta=0.1, epochs=50)

    # Test NN
    preds = []
    for i in xrange(X.shape[0]):
        pred = n.predict(X[i])
        preds.append(pred)
        print "{0} -> {1} ~ {2}:{3}".format(X[i], pred, c[i], y[i])

    print "Accuracy: {0}".format(accuracy(np.array(preds), c))


def iris_versicolor():
    # Input (X) and target (y) datasets
    X, y = split_data('iris-versicolor.txt')

    # Convert y into a binary matrix
    c = y_to_binary(y)

    # Create NN
    n = NN(X.shape[1], X.shape[1], c.shape[1])

    # Train NN
    n.train(X, c, eps=0.01, eta=0.1, epochs=50)

    # Test NN
    preds = []
    for i in xrange(X.shape[0]):
        pred = n.predict(X[i])
        preds.append(pred)
        print "{0} -> {1} ~ {2}:{3}".format(X[i], pred, c[i], y[i])

    print "Accuracy: {0}".format(accuracy(np.array(preds), c))


def XOR_NN():
    # Teach network XOR function
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # create a network with two input, two hidden, and one output nodes
    n = NN(X.shape[1], 2, y.shape[1])

    # epochs: 50 - 100, eta: 0.1 - 0.5
    n.train(X, y, eps=0.001, eta=0.4, epochs=500)

    # test it
    preds = []
    for i in xrange(X.shape[0]):
        pred = n.predict(X[i])
        preds.append(pred)
        print '{0} -> {1} ~ {2}'.format(X[i], pred, y[i])

    print accuracy(np.array(preds), y)
    # n.print_weights()


def AND_NN():
    # Teach network AND function
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0, 1],
                  [1, 0],
                  [1, 0],
                  [1, 0]])

    # create a network with two input, two hidden, and one output nodes
    n = NN(X.shape[1], 2, y.shape[1])

    # epochs: 50 - 100, eta: 0.1 - 0.5
    n.train(X, y, eps=0.001, eta=0.4, epochs=500)

    # test it
    preds = []
    for i in xrange(X.shape[0]):
        pred = n.predict(X[i])
        preds.append(pred)
        # print '{0} -> {1} ~ {2}'.format(X[i], pred, y[i])

    print accuracy(np.array(preds), y)
    # n.print_weights()


def script():
    filename = sys.argv[1]
    nh = int(sys.argv[2])
    eps = float(sys.argv[3])
    eta = float(sys.argv[4])
    epochs = int(sys.argv[5])

    # Input (X) and target (y) datasets
    X, y = split_data(filename)

    # Convert y into a binary matrix
    c = y_to_binary(y)

    # Create NN
    n = NN(X.shape[1], nh, c.shape[1])

    # Train NN
    n.train(X, c, eps=eps, eta=eta, epochs=epochs)

    # Test NN
    preds = []
    for i in xrange(X.shape[0]):
        pred = n.predict(X[i])
        preds.append(pred)
        print "{0} -> {1} ~ {2}:{3}".format(X[i], pred, c[i], y[i])

    acc = accuracy(np.array(preds), c)
    print "Accuracy: {0}".format(acc)

    with open("assign3-FUJIMOTO-JAMIE.txt", "w") as f:
        # Write inputs
        f.write("Dataset: {0}\n".format(filename))
        f.write("Hidden Neurons: {0}\n".format(nh))
        f.write("Epsilon: {0}\n".format(eps))
        f.write("Training Rate: {0}\n".format(eta))
        f.write("Epochs: {0}\n\n".format(epochs))

        # Write weights
        f.write("Input->Hidden Weights:\n{0}\n".format(n.wi))
        f.write("Hidden->Output Weights:\n{0}\n\n".format(n.wh))

        # Write accuracy
        f.write("Accuracy: {0}\n".format(acc * 100))


if __name__ == '__main__':
    # XOR_NN()
    # AND_NN()
    # iris()
    # iris_virginica()
    # iris_versicolor()
    script()  # run from command line