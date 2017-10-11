#%%
"""  Compute a BCE and run robust econometrics on observed bid distribution """
import sys
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import binom
import pandas as pd
sys.path.append(os.path.abspath('./lib'))
from bcemetricspy import bce_lp, utils


def plot_histograms(value_pdf, stats_file, fig_file):
    """ Plots duumped statistic data for mean identification.
    Parameters
    ----------
    - value_pdf: (array of float) the value pdf
    - stats_file: (string) file where statistics were dumped
    - fig_file: (string) where to store the histograms
    """

    stats = pd.DataFrame.from_csv(stats_file, sep='\t')

    fig = plt.figure(figsize=(25, 15))
    plt.subplot(231)
    plt.plot(list(range(len(value_pdf))), value_pdf)
    plt.title("PDF of values")
    plt.subplot(232)
    plt.hist(stats['revenue_list'])
    plt.title("Histogram of revenue lengths")
    plt.subplot(233)
    plt.hist(stats['easy_bound_list'])
    plt.title("Histogram easy upper bound")
    plt.subplot(234)
    plt.hist(stats['lb_mean_list'])
    plt.title("Histogram of mean lower bounds")
    plt.subplot(235)
    plt.hist(stats['ub_mean_list'])
    plt.title("Histogram of mean upper bounds")
    plt.subplot(236)
    plt.hist(stats['error_mean_list'])
    plt.title("Histogram of errors lengths")
    plt.tight_layout(pad=5)
    plt.savefig(fig_file)


def full_information_bce(num_bidders, max_value, value_pdf):
    """ BCE where players learn the true value and bid the true value. """
    bid_support = [tuple((v * np.ones(num_bidders)).astype(int))
                   for v in range(max_value + 1)]
    bid_pdf = {}
    for value in range(max_value + 1):
        bid_pdf[bid_support[value]] = value_pdf[value]

    return bid_pdf


def no_information_bce(num_bidders, max_value, value_pdf):
    """ Returns the BCE when players observe no information. """
    mean = np.dot(value_pdf, [utils.first_moment(v) for v in range(max_value + 1)])
    bid_support = tuple(np.round((mean * np.ones(num_bidders))).astype(int))
    bid_pdf = {bid_support: 1}
    return bid_pdf

def random_bce_mean_analysis(opts):
    """ Generates a distribution of values. Then computes many BCE from
    random objectives and for each BCE computes the identified set of the
    mean. Then plots histogram statistics of the sharp identified set.
    """
    num_bidders = opts.num_bidders
    max_value = opts.max_value

    # Common value distribution
    print("Generating value distribution.")
    value_pdf = {
        'binom': np.ndarray.flatten(
            binom(n=max_value, p=.5).pmf([list(range(0, max_value + 1))])),
        'random': np.ndarray.flatten(np.abs(np.random.standard_normal(max_value + 1))),
        'uniform': np.ones(max_value + 1)
    }.get(opts.value_pdf_type, np.abs(np.ndarray.flatten(np.random.standard_normal(max_value + 1))))
    value_pdf = value_pdf / np.sum(value_pdf)  # normalize just in case

    true_mean = np.dot(value_pdf, [utils.first_moment(v)
                                   for v in range(max_value + 1)])

    lb_mean_list = []
    ub_mean_list = []
    error_mean_list = []
    revenue_list = []
    easy_bound_list = []
    for bce in range(opts.num_bce):
        # Compute an observed bid distribution from some BCE
        print("Iteration {}: Computing BCE bid distribution.".format(bce))
        bid_pdf = bce_lp.compute_bce(num_bidders, max_value,
                                     value_pdf, num_samples=opts.num_samples,
                                     max_trials=opts.max_trials, random_seed=opts.random_seed + bce)

        # Check if we could not compute a BCE
        if bid_pdf is None:
            continue

        # Compute a sharp identified set on the mean and the variance
        print("Iteration {}: Doing econometrics on BCE.".format(bce))
        lb_mean, ub_mean = bce_lp.inverse_bce(
            num_bidders, max_value, bid_pdf, utils.first_moment, tolerance=opts.tolerance)

        revenue = sum([max(bids) * bid_pdf[bids] for bids in bid_pdf])
        easy_bound = utils.brooks_bound(num_bidders, max_value, bid_pdf)

        easy_bound_list.append(easy_bound)
        revenue_list.append(revenue)
        lb_mean_list.append(lb_mean)
        ub_mean_list.append(ub_mean)
        error_mean_list.append(max(ub_mean - true_mean, true_mean - lb_mean))

    stats_df = pd.DataFrame.from_dict({'easy_bound_list': easy_bound_list,
                                       'revenue_list': revenue_list, 'lb_mean_list': lb_mean_list,
                                       'ub_mean_list': ub_mean_list, 'error_mean_list': error_mean_list})

    stats_df.to_csv(opts.stats_file, sep='\t')

    plot_histograms(value_pdf, opts.stats_file, opts.fig_file)


def main(args):
    """
    Main Function. Example call:

    python many_bce_inference.py --num_bidders 2 --max_value 20 --num_samples 1000 --max_trials 30 --random_seed 12345 --tolerance 0.0001 --value_pdf_type uniform --plot 0 --num_bce 2 --stats_file stats.csv --fig_file stats.png

    INPUTS:
    - args: (list of strings) command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Robust identification in auctions! Example useage: \
                    python many_bce_inference.py --num_bidders 2 --max_value 20 --num_samples 1000 --max_trials 30 --random_seed 12345 --tolerance 0.0001 --value_pdf_type uniform --plot 0 --num_bce 2 --stats_file stats.csv --fig_file stats.png")
    parser.add_argument("--num_bidders", dest="num_bidders",
                        type=int, help='Number of bidders', default=2)
    parser.add_argument("--max_value", dest="max_value",
                        type=int, help='Maximum value/bid', default=4)
    parser.add_argument("--num_samples", dest="num_samples",
                        type=int, help='Bid support size', default=100)
    parser.add_argument("--max_trials", dest="max_trials",
                        type=int, help='Maximum trials for BCE computation', default=20)
    parser.add_argument("--random_seed", dest="random_seed",
                        type=int, help='Random seed', default=1234)
    parser.add_argument("--tolerance", dest="tolerance",
                        type=float, help='Tolerance in best response constraints', default=0.0001)
    parser.add_argument("--value_pdf_type", dest="value_pdf_type",
                        type=str, help='Type of value distribution. Can take any value in {binom, uniform, random}', default='binom')
    parser.add_argument("--num_bce", dest="num_bce",
                        type=int, help='How many BCE to compute and analyze', default=10)
    parser.add_argument("--stats_file", dest="stats_file",
                        type=str, help='A file where to store identification statistics for future use', default='stats.csv')
    parser.add_argument("--fig_file", dest="fig_file",
                        type=str, help='A figure filename that depicts histograms of statistics', default='stats.png')
    opts = parser.parse_args(args[1:])

    random_bce_mean_analysis(opts)


if __name__ == '__main__':
    main(sys.argv)
