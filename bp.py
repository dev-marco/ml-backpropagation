#!/usr/bin/env python3

import numpy as np


a = 1.0
ratio = 0.5
momentum = 0.0001

def sigm (x):
    return 1.0 / (1.0 + np.exp(-x * a))

def dsigm (x):
    return x * (1.0 - x)

def loss (expect, predict):
    return -(expect * np.log2(predict) + (1.0 - expect) * np.log2(1.0 - predict))

def dloss (inputs, expect, predict):
    # d/dw -y log(sigmoid(w x)) - (1 - y)(log(1 - sigmoid(w x)))
    return inputs * (1.0 - expect - predict)

def dloss (inputs, expect, predict):
    return expect - predict

def execute (inputs, weights):
    outputs = []

    for w in weights:
        inputs = np.array(sigm(np.dot(np.append(1.0, inputs), w)))
        outputs.append(inputs)

    return outputs

def gradients (inputs, expect, outputs, weights):
    grads = [ 0.0 ] * len(weights)

    grads[-1] = np.multiply(
        dloss(inputs, expect, outputs[-1]),
        dsigm(outputs[-1])
    )

    for i in range(len(weights) - 1):
        i = len(weights) - i - 2
        grads[i] = np.multiply(weights[i + 1][ 1 : ] * grads[i + 1], dsigm(outputs[i]))

    return grads

def update (inputs, grads, ratio, momentum, weights, olds = None):
    if olds is None:
        olds = [ np.zeros(w.shape, w.dtype) for w in weights ]

    result = []

    for weight, old, grad in enumerate(zip(weights, olds, grads)):
        result.append(weight + momentum * old + ratio * grad * inputs)


nodes = [ 2, 1 ]

inputs = np.array([ 0.1, 0.9 ])
weights = [
    np.array([
        [ 0.1, -0.2, 0.1 ],
        [ 0.1, -0.1, 0.3 ]
    ]).T,
    np.array(
        [ 0.2, 0.2, 0.3 ]
    ).T
]

if __name__ == '__main__':
    outputs = execute(inputs, *weights)
    grads = gradients(inputs, np.array([ 0.9 ]), outputs, weights)
