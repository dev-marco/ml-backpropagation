#!/usr/bin/env python3

import numpy as np
import argparse
import tempfile
import os

try:
    import matplotlib.pyplot as plt
except:
    plt = None


def to_pgm (img, expect):
    digit = np.argmax(expect)
    pixels = iter(img)

    with tempfile.NamedTemporaryFile('w', prefix = '{0}_'.format(digit), suffix = '.pgm', dir = '.', delete = False) as file:
        print('P2', file = file)
        print('28 28', file = file)
        print('255', file = file)

        for _ in range(28):
            for _ in range(28):
                print('{0}'.format(255 - int(next(pixels) * 255)), end = ' ', file = file)
            print(file = file)

        return file.name

def read_csv (fname):
    data = []
    digits = np.asmatrix(np.eye(10))

    with open(fname, 'r') as file:
        for line in file:
            img = map(int, line.split(','))
            expect = np.zeros(10)
            digit = next(img)

            data.append((
                np.asmatrix(np.fromiter(img, np.float) / 255.0).T,
                digits[ : , digit]
            ))

    return data

def sigm (x):
    with np.errstate(over = 'ignore'):
        return 1.0 / (1.0 + np.exp(-x))

def dsigm (x):
    return np.multiply(x, (1.0 - x))

def dcost (expect, predict):
    return predict - expect

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

def cost (data, weights):
    accum = 0

    for inp, expect in data:
        predict = execute(inp, weights)[-1]

        psum = predict.sum()

        if psum > 0:
            predict = predict / psum

        zero = np.asarray(expect == 0.0).reshape(-1)
        one = ~zero

        with np.errstate(divide = 'ignore'):
            predict[zero] = np.nan_to_num(np.log(1 - predict[zero]))
            predict[one] = np.multiply(expect[one], np.log(predict[one]))

        accum -= predict.sum()

    return accum / len(data)

def error (data, weights):
    count = 0

    for inp, expect in data:

        if np.argmax(expect) != np.argmax(execute(inp, weights)[-1]):
            count += 1

    return count / len(data)

argparser = argparse.ArgumentParser()

argparser.add_argument('input', type = str)
argparser.add_argument('-momentum', type = float, default = 0.0001)
argparser.add_argument('-ratio', type = float, default = [], nargs = '*')
argparser.add_argument('-batch', type = float, default = [], nargs = '*')
argparser.add_argument('-hidden', type = int, default = [], nargs = '*')
argparser.add_argument('-generations', type = float, default = np.inf)
argparser.add_argument('-stop', type = float, default = -np.inf)
argparser.add_argument('-validate', type = str, default = False)
argparser.add_argument('-save', type = str, default = False)
argparser.add_argument('-no-dump', action = 'store_false', dest = 'dump')
argparser.add_argument('-plot', type = str, default = False)
argparser.add_argument('-threads', type = int, default = 1)

argparser.add_argument('-fix-ratio', type = float, default = 0.1)
argparser.add_argument('-fix-batch', type = float, default = 10.0)
argparser.add_argument('-fix-hidden', type = int, default = 100)

args = argparser.parse_args()

params = [
    ( args.ratio or [ args.fix_ratio ], args.fix_ratio ),
    ( args.batch or [ args.fix_batch ], args.fix_batch ),
    ( args.hidden or [ args.fix_hidden ], args.fix_hidden )
]

experiments = []

if all(map(lambda x: len(x[0]) == 1, params)):
    experiments.append(tuple(p[0][0] for p in params))
else:
    for i, p in enumerate(params):
        for v in p[0]:
            experiments.append(
                tuple(v if j == i else fix[1] for j, fix in enumerate(params))
            )

unique = set()
experiments = [ e for e in experiments if not (e in unique or unique.add(e)) ]

print('reading files')
train = read_csv(args.input)
validate = read_csv(args.validate) if args.validate else train

print('{0} train instances'.format(len(train)))

if args.validate:
    print('{0} validation instances'.format(len(validate)))

for ratio, batch, hidden in experiments:

    print('\nratio = {0}, batch = {1}, hidden = {2}'.format(ratio, batch, hidden))

    train_errors = []
    validate_errors = []

    nodes = ( 784, hidden, 10 )

    old = weights = [
        np.asmatrix(np.random.randn(nf + 1, nt))
            for nf, nt in zip(nodes[ : -1 ], nodes[ 1 : ])
    ]

    i = 0
    while i < args.generations:
        i += 1

        train_error = error(train, weights)
        validate_error = (
            error(validate, weights)
                if args.validate else
            train_error
        )

        if validate_error <= args.stop:
            break

        train_cost = np.linalg.norm(cost(train, weights))
        validate_cost = (
            np.linalg.norm(cost(validate, weights))
                if args.validate else
            train_cost
        )

        if args.validate:
            validate_errors.append(validate_error)

        train_errors.append(train_error)

        if args.dump:
            print('generation {0}, validation = {1:.5f}, train = {2:.5f}, cost = {3:.5f}'.format(
                i, validate_error, train_error, validate_cost
            ))

        batch_size = 0
        accum = [ np.zeros(w.shape, w.dtype) for w in weights ]

        np.random.shuffle(train)

        for j, (inp, expect) in enumerate(train):
            out = execute(inp, weights)
            grads = gradients(expect, out, weights)

            accum = np.add(accum, delta(inp, grads, out))

            batch_size += 1

            if batch_size == batch or (j + 1) == len(train):

                new_weights = np.add(weights, np.subtract(
                    np.multiply(args.momentum, old),
                    np.multiply(ratio, np.divide(accum, batch_size))
                ))

                old = weights
                weights = new_weights

                batch_size = 0
                accum = [ np.zeros(w.shape, w.dtype) for w in weights ]

    if plt and args.plot:
        y_axis = np.arange(1, len(train_errors) + 1, dtype = np.uint)

        plt.plot(y_axis, train_errors, 'r-', label = 'Train')

        if args.validate:
            plt.plot(y_axis, validate_errors, 'b-', label = 'Validation')
            plt.legend()

        plt.xlabel('Generation')
        plt.ylabel('Error')

        fname = os.path.join(
            args.plot, '{0}-{1}-{2}.pdf'.format(ratio, batch, hidden)
        )

        plt.savefig(fname, dpi = 300, transparent = True)

    if args.save:
        fname = os.path.join(
            args.save, '{0}-{1}-{2}.txt'.format(ratio, batch, hidden)
        )

        with open(fname, 'w') as file:
            print(' '.join(map(str, train_errors)), file = file)

            if args.validate:
                print(' '.join(map(str, validate_errors)), file = file)

print('\ndone')
