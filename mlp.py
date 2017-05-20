#!/usr/bin/env python3

import os
import time
import signal
import argparse
import tempfile
import multiprocessing
import itertools as it
import numpy as np


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

        with np.errstate(divide = 'ignore', over = 'ignore'):
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

mlp_globals = {}

def mlp (data):
    ( ratio, batch, hidden ), pos, size = data

    lock = mlp_globals['lock']
    train = mlp_globals['train']
    validate = mlp_globals['validate']
    args = mlp_globals['args']

    if batch != np.inf:
        batch = int(batch)

    generations = args.generations if args.generations == np.inf else int(args.generations)

    if args.dump:
        with lock:
            print('( ratio = {}, batch = {}, hidden = {}, pos = {} / {} ): STARTING'.format(
                ratio, batch, hidden, pos + 1, size
            ), flush = True)

    train_errors = []
    validate_errors = []

    nodes = ( 784, hidden, 10 )

    old = weights = None
    gen = 0

    if args.load:
        fname = os.path.join(
            args.load, '{0}-{1}-{2}-weights.txt'.format(ratio, batch, hidden)
        )

        if os.path.isfile(fname):
            with open(fname, 'r') as file:
                gen = int(next(file))
                size = int(next(file))

                weights = [ None ] * size
                old = [ None ] * size

                for i in range(size):
                    rows, cols = map(int, next(file).strip().split())
                    dtype = next(file).strip()

                    weights[i] = np.asmatrix(
                        np.fromiter(map(float, next(file).strip().split()), dtype)
                    ).reshape(rows, cols)


                    old[i] = np.asmatrix(
                        np.fromiter(map(float, next(file).strip().split()), dtype)
                    ).reshape(rows, cols)

    if weights is None:
        old = weights = [
            np.asmatrix(np.random.randn(nf + 1, nt))
                for nf, nt in zip(nodes[ : -1 ], nodes[ 1 : ])
        ]

    start_time = time.time()

    try:
        while gen < generations:

            gen += 1

            train_error = error(train, weights)
            validate_error = (
                error(validate, weights)
                    if args.validate else
                train_error
            )

            if args.validate:
                validate_errors.append(validate_error)

            train_errors.append(train_error)

            train_cost = np.linalg.norm(cost(train, weights))
            validate_cost = (
                np.linalg.norm(cost(validate, weights))
                    if args.validate else
                train_cost
            )

            if train_error <= args.stop:
                break

            if args.dump:
                with lock:
                    print('( ratio = {}, batch = {}, hidden = {}, pos = {} / {} ): ITERATION\n  '
                          'gen {} / {} verr = {:.5f} terr = {:.5f} '
                          'vcost = {:.5f} tcost = {:.5f}'.format(
                        ratio, batch, hidden, pos + 1, size,
                        gen, generations, validate_error, train_error,
                        validate_cost, train_cost
                    ), flush = True)

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

    except KeyboardInterrupt:
        pass

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if train_errors:
        fname = os.path.join(
            args.output, '{}-{}-{}.txt'.format(ratio, batch, hidden)
        )

        with open(fname, 'w') as file:
            print(' '.join(map(str, train_errors)), file = file)

            if args.validate:
                print(' '.join(map(str, validate_errors)), file = file)

        if args.save:
            fname = os.path.join(
                args.save, '{}-{}-{}-weights.txt'.format(ratio, batch, hidden)
            )

            with open(fname, 'w') as file:
                print(gen, file = file)
                print(len(weights), file = file)

                for w, o in zip(weights, old):
                    print('{} {}'.format(w.shape[0], w.shape[1]), file = file)
                    print('{}'.format(w.dtype.char), file = file)
                    print(' '.join(map(str, np.ravel(w))), file = file)
                    print(' '.join(map(str, np.ravel(o))), file = file)

    if args.dump:
        with lock:
            print('( ratio = {}, batch = {}, hidden = {}, pos = {} / {} ): ENDING\n  '
                  'time = {:.5f}s generations = {}'.format(
                ratio, batch, hidden, pos + 1, size,
                time.time() - start_time, gen
            ), flush = True)

argparser = argparse.ArgumentParser()

argparser.add_argument('input', type = str)
argparser.add_argument('output', type = str)
argparser.add_argument('-momentum', type = float, default = 0.0001)
argparser.add_argument('-ratio', type = float, default = [ 0.5 ], nargs = '*')
argparser.add_argument('-batch', type = float, default = [ 10.0 ], nargs = '*')
argparser.add_argument('-hidden', type = int, default = [ 100 ], nargs = '*')
argparser.add_argument('-generations', type = float, default = np.inf)
argparser.add_argument('-stop', type = float, default = -np.inf)
argparser.add_argument('-validate', type = str, default = False)
argparser.add_argument('-save', type = str, default = False)
argparser.add_argument('-load', type = str, default = False)
argparser.add_argument('-no-dump', action = 'store_false', dest = 'dump')
argparser.add_argument('-threads', type = int, default = multiprocessing.cpu_count())

args = argparser.parse_args()

print('reading files', flush = True)
train = validate = read_csv(args.input)
print('{0} train instances'.format(len(train)), flush = True)

if args.validate:
    validate = read_csv(args.validate)
    print('{0} validation instances'.format(len(validate)), flush = True)

mlp_globals = {
    'args': args,
    'train': train,
    'validate': validate,
    'lock': multiprocessing.Lock()
}

unique = set()
experiments = [
    e for e in it.product(*( args.ratio, args.batch, args.hidden ))
        if not (e in unique or unique.add(e))
]

pool = multiprocessing.Pool(min(args.threads, len(experiments)))
asyn = pool.map_async(mlp, (
    ( e, pos, len(experiments) )
        for pos, e in enumerate(experiments)
), 1)

sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

asyn.wait()

signal.signal(signal.SIGINT, sigint)

pool.close()
pool.join()

print('\ndone', flush = True)
