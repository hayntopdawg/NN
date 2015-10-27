import csv
import numpy as np

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


# Used for output layer (N_o)
def get_num_classes(y):
    return len(set(y))


# Converts y into a binary matrix
def convert_classes(y):
    c_matrix = np.zeros([y.shape[0], get_num_classes(y)])
    classes = list(set(y))
    for i, c in enumerate(y):
        c_matrix[i, classes.index(c)] = 1
    return c_matrix


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NN():
    # Assume only one hidden layer of nh neurons
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output neurons
        self.ni = ni + 1 # + 1 for the bias node
        self.nh = nh + 1 # + 1 for the bias node
        self.no = no

        # initialize activations
        self.ai = np.ones((self.ni, 1))
        # print 'ai: {0}'.format(self.ai.shape)
        self.ah = np.ones((self.nh, 1))
        # print 'ah: {0}'.format(self.ah.shape)
        self.ao = np.ones((self.no, 1))
        # print 'ao: {0}'.format(self.ao.shape)

        # seed random numbers to make calculation deterministic
        np.random.seed(1)

        # initialize weights with random values between -0.5 and 0.5
        self.wi = np.random.rand(self.ni, self.nh - 1)  # ni x nh matrix
        # print 'wi: {0}'.format(self.wi)
        self.wh = np.random.rand(self.nh, self.no)  # nh x no matrix
        # print 'wh: {0}'.format(self.wh)


    def forward_feed(self, Xi):
        # input activations
        self.ai = np.append(Xi, 1)  # add bias node back
        # print 'ai: {0}'.format(self.ai)

        # hidden activations
        ah = sigmoid(np.dot(self.ai.T, self.wi).T)
        self.ah = np.append(ah, 1)  # add bias node back
        # print 'ah: {0}'.format(self.ah)

        # output activations
        self.ao = sigmoid(np.dot(self.ah.T, self.wh).T)
        # print 'ao: {0}'.format(self.ao)


    def back_propagate(self):
        pass


    def train(self):
        pass


    def test(self):
        pass


    def print_weights(self):
        pass


def iris():
    # Input (X) and target (y) datasets
    X, y = split_data('iris.data.txt')
    # Convert y into a binary matrix
    c = convert_classes(y)
    # Input layer is X_i


def demo():
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
    n = NN(2, 2, 1)

    n.forward_feed(X[0])
    # # train it with some patterns
    # n.train(pat)
    # # test it
    # n.test(pat)


if __name__ == '__main__':
    demo()