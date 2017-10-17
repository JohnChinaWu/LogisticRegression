import math
import numpy as np
#-------------------------------------------------------------------
def log(n):
    return math.log(n)
#-------------------------------------------------------------------
def exp(n):
    return math.exp(n)
#-------------------------------------------------------------------
class logistic:
    #******************************************************
    def __init__(self, parameters):
        self.parameters = parameters
        data = np.array([[60, 155], [64, 135], [73, 170]])
        data = (data - data.mean(0)) / data.std(0)
        self.features = np.hstack((np.ones((data.shape[0], 1)), data))
        self.labels = np.array([0, 1, 1])
    #******************************************************
    ########## Feel Free to Add Helper Functions ##########
    #******************************************************
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def log_likelihood(self):
        ll = 0.0
        ##################### Please Fill Missing Lines Here #####################
        scores = np.dot(self.features, self.parameters)
        ll = np.sum(self.labels * scores - np.log(1 + np.exp(scores)))
        return ll
    #******************************************************
    def gradients(self):
        gradients = []
        ##################### Please Fill Missing Lines Here #####################
        predic = self.sigmoid(np.dot(self.features, self.parameters))
        gradients = np.dot(self.features.T, self.labels - predic)
        return gradients
    #******************************************************
    def iterate(self):
        ##################### Please Fill Missing Lines Here #####################
        # print 'gradient'
        # print self.gradients()
        # print 'hessian'
        # print self.hessian()
        self.parameters -= np.dot(np.linalg.inv(self.hessian()), self.gradients())
        return self.parameters
    #******************************************************
    def hessian(self):
        n = len(self.parameters)
        hessian = np.zeros((n, n))
        ##################### Please Fill Missing Lines Here #####################
        p = self.sigmoid(np.dot(self.features, self.parameters))
        p *= 1 - p
        hessian = -np.dot(self.features.T * p, self.features)
        return hessian
#-------------------------------------------------------------------
parameters = []
##################### Please Fill Missing Lines Here #####################
## initialize parameters
parameters = np.ones(3) * .25
# print parameters
l = logistic(parameters)
parameters = l.iterate()
# print 'parameters:'
# print parameters
l = logistic(parameters)
parameters = l.iterate()
# print 'parameters:'
print parameters
