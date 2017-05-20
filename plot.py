#!/usr/bin/env python3

import os
import re
import argparse
import itertools as it
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import unidecode


colors = it.cycle([
    'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
    '#0504aa', '#a9f971'
])

def sanitize (text, _re = re.compile(r'[^a-zA-Z0-9.-]+')):
    return _re.sub('_', unidecode.unidecode(text)).lower().strip('_-.')

argparser = argparse.ArgumentParser()

argparser.add_argument('inputs', type = str, nargs = '*')
argparser.add_argument('-labels', type = str, default = [], nargs = '*')
argparser.add_argument('-title', default = 'chart', type = str)
argparser.add_argument('-validate', action = 'store_true')
argparser.add_argument('-folder', type = str, default = '.')
argparser.add_argument('-legend-size', type = float, default = 9.0)
argparser.add_argument('-title-size', type = float, default = 12.0)
argparser.add_argument('-lang', type = str, default = 'en')

args = argparser.parse_args()

data = []

for fname, label in zip(args.inputs, args.labels):
    with open(fname, 'r') as file:

        if args.validate:
            data.append((
                label, (
                    np.fromiter(map(float, next(file).split(' ')), np.float),
                    np.fromiter(map(float, next(file).split(' ')), np.float)
                )
            ))
        else:
            data.append((
                label,
                np.fromiter(map(float, next(file).split(' ')), np.float)
            ))

mpl.rcParams['savefig.bbox'] = 'tight'

def save (args, name):
    plt.ylim(( 0.0, 1.0 ))
    plt.yticks(np.arange(0.1, 1.0, 0.1))

    plt.title(name, fontsize = args.title_size)

    plt.legend(fontsize = args.legend_size, loc='center left', bbox_to_anchor = ( 1, 0.5 ))

    plt.xlabel('Geração' if args.lang == 'br' else 'Generation')
    plt.ylabel('Erro' if args.lang == 'br' else 'Error')

    plt.minorticks_on()

    plt.gca().yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    plt.gca().yaxis.grid(True, which = 'major', linestyle = ':', color = '0.5')
    plt.gca().yaxis.grid(True, which = 'minor', linestyle = ':', color = '0.7')

    plt.savefig(os.path.join(
        args.folder, '{0}.pdf'.format(sanitize(name))
    ), dpi = 300, transparent = True)


if args.validate:
    for label, ( train, validate ) in data:
        fig = plt.figure()

        y_axis = np.arange(1, len(train) + 1, dtype = np.uint)

        plt.plot(y_axis, train, label = '$E_{in}$')
        plt.plot(y_axis, validate, label = '$E_{out}$', linewidth = 0.5, color = '#FF0000A5')

        save(args, label)

else:
    fig = plt.figure()

    for label, train in data:
        y_axis = np.arange(1, len(train) + 1, dtype = np.uint)
        plt.plot(y_axis, train, label = label, color = next(colors), linewidth = 1.0)

    save(args, args.title)
