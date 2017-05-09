#!/usr/bin/env python3

import numpy as np
import sys


def sigm (x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigm (x):
    return np.multiply(x, (1.0 - x))

def cost (expect, predict):
    return np.nan_to_num(-np.multiply(expect, np.log(predict)) - np.multiply((1 - expect), np.log(1 - predict))).sum()
    # return -(
    #     np.multiply(expect, np.log(predict + np.finfo(np.float64).tiny)) +
    #     np.multiply(1.0 - expect, np.log(1.0 - predict + np.finfo(np.float64).tiny))
    # )

def dcost (expect, predict):
    # d/dw -y log(f(w)) - (1 - y)(log(1 - f(w)))
    return predict - expect
    # d/dw -y log(sigmoid(w x)) - (1 - y)(log(1 - sigmoid(w x)))
    # return inputs * (1.0 - expect - predict)

def execute (inp, weights):
    result = []

    for w in weights:
        inp = sigm(np.dot(np.append(1.0, inp), w).T)
        result.append(inp)

    return result

def gradients (expect, out, weights):
    result = [ 0.0 ] * len(weights)

    result[-1] = dcost(expect, out[-1])

    for i in range(len(weights) - 1):
        i = len(weights) - i - 2
        result[i] = weights[i + 1][ 1 : ] * result[i + 1]

    return result

def delta (inp, grad, out):
    result = []

    for g, o in zip(grad, out):
        result.append(np.outer(np.append(1.0, inp), g))
        inp = o

    return result

def costs (data, weights):
    accum = None

    for inp, expect in data:
        if accum is None:
            accum = cost(expect, execute(inp, weights)[-1])
        else:
            accum += cost(expect, execute(inp, weights)[-1])

    return accum / len(data)

def error (data, weights):
    errors = 0

    for inp, expect in data:

        if np.argmax(expect) != np.argmax(execute(inp, weights)[-1]):
            errors += 1

    return errors / len(data)


ratio = 0.1
momentum = 0.0001
batch_size = 10
generations = 30

nodes = [ 784, 100, 10 ]

data = []
digits = {}

digit_vectors = np.asmatrix(np.eye(10))

with open(sys.argv[1], 'r') as file:
    for line in file:
        img = map(int, line.split(','))
        expect = np.zeros(10)
        digit = next(img)

        pixels = np.asmatrix(np.fromiter(img, np.float) / 255.0).T

        train = ( pixels, digit_vectors[ : , digit] )

        data.append(train)
        digits.get(digit, []).append(pixels)

weights = [
    np.asmatrix(np.random.randn(nf + 1, nt))
        for nf, nt in zip(nodes[ : -1 ], nodes[ 1 : ])
]

test_size = int(len(data) * 0.95)

if __name__ == '__main__':

    old = weights

    try:
        for i in range(generations):

            if not (i % max(generations / 50, 1)):
                print('generation {0}, error = {1}'.format(i, error(data, weights)))

            batch = 0
            accum = [ np.zeros(w.shape, w.dtype) for w in weights ]

            np.random.shuffle(data)
            testing = data[ : test_size ]

            for j, (inp, expect) in enumerate(testing):
                out = execute(inp, weights)
                grads = gradients(expect, out, weights)

                accum = np.add(accum, delta(inp, grads, out))

                batch += 1

                if batch == batch_size or (j + 1) == len(testing):

                    new_weights = np.add(weights, np.subtract(
                        np.multiply(momentum, old),
                        np.multiply(ratio, np.divide(accum, batch))
                    ))

                    old = weights
                    weights = new_weights

                    batch = 0
                    accum = [ np.zeros(w.shape, w.dtype) for w in weights ]

    except KeyboardInterrupt:
        print()
        pass

    print(error(data, weights))

    # inp, expect = validation[np.random.randint(0, len(validation))]
    #
    # with open('output.pgm', 'w') as img:
    #     print('P2', file = img)
    #     print('28 28', file = img)
    #     print('255', file = img)
    #
    #     pixels = iter(inp)
    #
    #     for i in range(28):
    #         for j in range(28):
    #             print('{0}'.format(int(255 * (1.0 - float(next(pixels))))), end = ' ', file = img)
    #         print(file = img)
    #
    # print(execute(inp, weights)[-1])
    # print(expect)
