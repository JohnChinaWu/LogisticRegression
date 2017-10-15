import math
import timeit

import numpy as np
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------------
def log(n):
    return math.log(n)


# -------------------------------------------------------------------
def exp(n):
    return math.exp(n)


# -------------------------------------------------------------------
class logistic:
    # ******************************************************
    def __init__(self, parameters):
        self.nums = 5000
        self.alpha = 0.001
        np.random.seed(12)
        if len(parameters) == 0:
            self.parameters = np.zeros(3)
        else:
            self.parameters = np.array(parameters)
        self.x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], self.nums)
        self.x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], self.nums)
        self.features = np.hstack((np.ones([self.nums * 2, 1]), np.vstack((self.x1, self.x2)).astype(np.float32)))
        self.labels = np.hstack((np.zeros(self.nums), np.ones(self.nums)))

    # ******************************************************
    ########## Feel Free to Add Helper Functions ##########
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def bench_mark(self):
        print '------------------------------------------------'
        print 'Logistic Regression\'s bench mark'
        g = LogisticRegression(fit_intercept=True, C=1e15)
        g.fit(self.features, self.labels)
        print 'Accuracy: {0}'.format(g.score(self.features, self.labels))

    def test_gradient(self):
        print '------------------------------------------------'
        print 'Logistic Regression using gradient'
        for i in xrange(5000):
            self.iterate()

            if i % 1000 == 0:
                print 'Likelihood (iterations {0}): {1}'.format(i, self.log_likelihood())
        p = np.round(self.sigmoid(np.dot(self.features, self.parameters)))
        print 'Accuracy: {0}'.format((p == self.labels).sum().astype(np.float32) / len(p))

    def test_hessian(self):
        print '------------------------------------------------'
        print 'Logistic Regression using hessian'
        for i in xrange(10):
            self.iterate_hessian()
            if i % 2 == 0:
                print 'Likelihood (iterations {0}): {1}'.format(i, self.log_likelihood())
        p = np.round(self.sigmoid(np.dot(self.features, self.parameters)))
        print 'Accuracy: {0}'.format((p == self.labels).sum().astype(np.float32) / len(p))

    def iterate_hessian(self):
        self.parameters -= np.dot(np.linalg.inv(self.hessian()), self.gradients())
        return self.parameters

    # ******************************************************
    def log_likelihood(self):
        ll = 0.0
        ##################### Please Fill Missing Lines Here #####################
        scores = np.dot(self.features, self.parameters)
        ll = np.sum(self.labels * scores - np.log(1 + np.exp(scores)))
        return ll

    # ******************************************************
    def gradients(self):
        gradients = []
        ##################### Please Fill Missing Lines Here #####################

        predit = self.sigmoid(np.dot(self.features, self.parameters))
        gradients = np.dot(self.features.T, self.labels - predit)
        return gradients

    # ******************************************************
    def iterate(self):
        ##################### Please Fill Missing Lines Here #####################

        self.parameters += self.alpha * self.gradients()
        return self.parameters

    # ******************************************************
    def hessian(self):
        n = len(self.parameters)
        hessian = np.zeros((n, n))
        ##################### Please Fill Missing Lines Here #####################
        p = self.sigmoid(np.dot(self.features, self.parameters))
        p *= 1 - p
        hessian = -np.dot(self.features.T * p, self.features)

        return hessian


# -------------------------------------------------------------------
parameters = []
##################### Please Fill Missing Lines Here #####################
## initialize parameters

start = timeit.default_timer()
logistic(parameters).test_gradient()
print 'time: %ss\n' % (timeit.default_timer() - start)

start = timeit.default_timer()
logistic(parameters).test_hessian()
print 'time: %ss\n' % (timeit.default_timer() - start)

start = timeit.default_timer()
logistic(parameters).bench_mark()
print 'time: %ss\n' % (timeit.default_timer() - start)

# l = logistic(parameters)
# parameters = l.iterate()
# l = logistic(parameters)
# print l.iterate()
