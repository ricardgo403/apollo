from os.path import isfile, dirname, join, realpath
import numpy as np
from os import listdir, mkdir
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize


def main(v):
    def fit(t, A, b, c, d):
        return A * np.cos(b * t + c) + d

    df = pd.read_csv(join(dirname(realpath(__file__)), join(v, f'{v}.csv')))
    p0 = [100, 5, np.pi, 450]
    net = optimize.curve_fit(fit, xdata=df.t[0:100], ydata=df.y[0:100], p0=p0)
    t = np.arange(0, 20, 0.01)
    plt.plot(t, fit(t, *net[0]), color='black')
    plt.scatter(df.t, df.y, s=2, color='red')
    plt.savefig(join(dirname(realpath(__file__)), join(v, f'{v}cos.png')), dpi=500, bbox_inches='tight', pad_inches=0.5)
    plt.show()

    def fit2(t, A, w_i, w_e, b, c):
        return np.exp(-w_e * t) * A * np.cos(((w_i - w_e ** 2) ** 0.5) * t + b) + c

    p0 = [100, 5, 0.1, np.pi, 450]
    net = optimize.curve_fit(fit2, xdata=df.t, ydata=df.y, p0=p0)
    t = np.arange(0, 20, 0.01)
    plt.plot(t, fit2(t, *net[0]), color='black')
    plt.scatter(df.t, df.y, s=1, color='red')
    df = pd.DataFrame(net[0], index=['Amp', 'w_r', 'w_d', 'desf', 'h'], columns=None)
    df.to_csv(join(dirname(realpath(__file__)), join(v, f'fit.csv')))
    plt.savefig(join(dirname(realpath(__file__)), join(v, f'{v}dump.png')), dpi=500, bbox_inches='tight',
                pad_inches=0.5)

    plt.show()
