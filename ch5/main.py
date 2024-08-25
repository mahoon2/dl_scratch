import sys, os
sys.path.append(os.pardir)

import numpy as np

from mnist import load_mnist
from collections import OrderedDict

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        return self.x * self.y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return (dx, dy)

class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x, y):
        return x + y
    
    def backward(self, dout):
        return (dout, dout)

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
    
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x

    def loss(self, x, t):
        ret = self.predict(x)
        return self.lastLayer.forward(ret, t)

    def accuracy(self, x, t):
        ret = self.predict(x)
        ret = np.argmax(ret, axis = 1)
        if t.ndim != 1:
            t = np.argmax(t, axis = 1)
    
        return np.sum(ret == t) / float(x.shape[0])
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = dict()
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        ret = self.lastLayer.backward()
        for layer in reversed(self.layers.values()):
            ret = layer.backward(ret)

        grads = dict()
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))

def relu(x: np.array) -> np.array:
    return np.maximum(0, x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis = 0)
        y = np.exp(x) / np.sum(np.exp(x), axis = 0)
        return y.T 

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y: np.array, t: np.array):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if t.size == y.size:
        t = t.argmax(axis = 1)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()   
        
    return grad

def main():
    # # Forward
    # apple_price = 100
    # apple_num   = 2
    
    # orange_price = 150
    # orange_num   = 3

    # tax = 1.1

    # apple_MulLayer  = MulLayer()
    # orange_MulLayer = MulLayer()
    # fruit_AddLayer  = AddLayer()
    # tax_MulLayer    = MulLayer()

    # apple_MulOut  = apple_MulLayer.forward(apple_num, apple_price)
    # orange_MulOut = orange_MulLayer.forward(orange_num, orange_price)
    # fruit_AddOut  = fruit_AddLayer.forward(apple_MulOut, orange_MulOut)
    # ret           = tax_MulLayer.forward(fruit_AddOut, tax)
    # print(ret)

    # # Backward
    # dprice = 1
    # dfruit, dtax = tax_MulLayer.backward(dprice)
    # dapple, dorange = fruit_AddLayer.backward(dfruit)
    # dapple_num, dapple_price = apple_MulLayer.backward(dapple)
    # dorange_num, dorange_price = orange_MulLayer.backward(dorange)
    # print(dapple_num, dapple_price, dorange_num, dorange_price, dtax)

    (x_train, t_train), (x_test, t_test) = load_mnist(True, True, True)
    network = TwoLayerNet(784, 50, 10)

    x_batch = x_train[:3]
    t_batch = t_train[:3]

    grad_num  = network.numerical_gradient(x_batch, t_batch)
    grad_back = network.gradient(x_batch, t_batch)

    for key in grad_num.keys():
        diff = np.average(np.abs(grad_back[key] - grad_num[key]))
        print(key + ":" + str(diff))

if __name__ == '__main__':
    main()