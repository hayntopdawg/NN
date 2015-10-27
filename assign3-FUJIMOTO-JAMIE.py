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


    def forward_feed(self, xi):
        # input activations
        self.ai = np.append(xi, 1)  # add bias node back
        # print 'ai ({0}): {1}'.format(self.ai.shape, self.ai)

        # print 'wi: {0}'.format(self.wi)

        # hidden activations
        ah = sigmoid(np.dot(self.wi.T, self.ai))
        self.ah = np.append(ah, 1)  # add bias node back
        # print 'ah ({0}): {1}'.format(self.ah.shape, self.ah)

        # print 'wh: {0}'.format(self.wh)

        # output activations
        self.ao = sigmoid(np.dot(self.wh.T, self.ah))
        # print 'ao: {0}'.format(self.ao)
        return self.ao


    def back_propagate(self, xi, yi, eta):
        # calculate error terms for outputs
        djo = (yi - self.ao) * self.ao * (1 - self.ao)
        # print 'djo: {0}'.format(djo)

        # calculate error terms for hidden terms
        djh = self.ah * (1 - self.ah) * np.dot(djo, self.wh.T)
        # print 'djh: {0}'.format(djh)
        # print 'wh: {0}'.format(self.wh)

        # update hidden weights
        self.wh = self.wh + (eta * djo * self.ao)
        # print 'wh: {0}'.format(self.wh)

        # update input weights
        self.wi = (self.wi.T + (eta * djh * self.ao)).T
        # print 'wi: {0}'.format(self.wi)


    def train(self, X, y, eps, eta, epochs):
        for e in xrange(epochs):
            r = np.arange(X.shape[0])
            # print 'r: {0}'.format(r)
            np.random.shuffle(r)
            # print 'r: {0}'.format(r)
            for i in r:
                converged = False
                while not converged:
                    # print 'Converging'
                    self.forward_feed(X[i])
                    E = 0.5 * (self.ao - y[i])**2
                    # print 'E[{0}]: {1}'.format(i, E)
                    if E <= eps:
                        converged = True
                    else:
                        # print 'E > eps'
                        self.back_propagate(X[i], y[i], eta)


    def test(self, X):
        for i in xrange(X.shape[0]):
            print '{0} -> {1}'.format(X[i], self.forward_feed(X[i]))


    def print_weights(self):
        print self.wi
        print self.wh


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
    n = NN(X.shape[1], 2, y.shape[1])

    n.forward_feed(X[0])

    # epochs: 50 - 100
    # n.train(X, y, eps=0.01, eta=0.5, epochs=100)
    # # test it
    # n.test(X)
    # n.print_weights()


if __name__ == '__main__':
    demo()