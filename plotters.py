"""Some helper functions for machine learning research

Code for many of these routines was taken from *Python Machine Learning - Second Edition*, by Raschka and Mirjalili
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve



def plot_means_w_stds(means, stds, xrange, series_labels=None, ylabel=None, xlabel=None, legend=True, linestyles=None, title=None, ylim=None, logx=False, legend_kwargs={}, fig_kwargs={}, markersize=None):
    """Generic plot routine to plot multiple lines on same axes"""

    fig, ax = plt.subplots(**fig_kwargs)

    if not linestyles:
        if len(means) % 2 == 1:
            # ODD sequence
            linestyles = ['-'] * int(len(means))
        else:
            linestyles = ['-'] * int(len(means) / 2) + ['--'] * int(len(means) / 2)
    if not markersize:
        markersize = 5
    if not series_labels:
        series_labels = [''] * len(means)

    for ix, mean, std, label, ls in zip(range(len(means)), means, stds, series_labels, linestyles):
        color = 'C%s' % ix
        plt.plot(xrange, mean, marker='o', markersize=markersize, label=label, color=color, linestyle=ls)
        plt.fill_between(xrange, mean + std, mean - std, color=color, alpha=0.15)

    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    if logx:
        plt.semilogx()
    if legend:
        plt.legend(**legend_kwargs)
    if title:
        plt.title(title)
    plt.show()


def gen_and_plot_learning_curve(estimator, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), ylim=None, ylabel=None, title=None, **kwargs):
    """Plot a learning curve for given estimator"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X_train,
        y=y_train,
        train_sizes=train_sizes,
        **kwargs
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std, ylim, ylabel, title)

    return train_sizes, train_mean, train_std, test_mean, test_std


def plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std, ylim=None, ylabel=None, title=None):


    plt.plot(train_sizes, train_mean,
             color='C0', marker='o',
             markersize=5, label='training')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='C0')

    plt.plot(train_sizes, test_mean,
             color='C1', linestyle='--',
             marker='s', markersize=5,
             label='validation')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='C1')
    plt.grid()
    plt.xlabel('Number of training samples')
    if not ylabel:
        plt.ylabel('Accuracy')
    else:
        plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.ylim(ylim)
    plt.show()

