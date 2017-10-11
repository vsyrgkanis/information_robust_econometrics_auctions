#%%
"""  Compute a BCE and run robust econometrics on observed bid distribution """
import sys
import os
import argparse
import math
import numpy as np
from scipy.stats import binom
sys.path.append(os.path.abspath('./lib'))
from bcemetricspy import bce_lp, utils

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

def mean_and_variance_analysis(opts):
    """ Generates a distribution of values, computes a BCE and then
    computes the sharp identified set of the mean and the second
    order.
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

    # Compute an observed bid distribution from some BCE
    print("Computing BCE bid distribution.")
    bid_pdf = bce_lp.compute_bce(num_bidders, max_value,
                                 value_pdf, num_samples=opts.num_samples,
                                 max_trials=opts.max_trials, random_seed=opts.random_seed)

    # Check if we could not compute a BCE
    if bid_pdf is None:
        return

    # bid_pdf = full_information_bce(num_bidders, max_value, value_pdf)
    # bid_pdf = no_information_bce(num_bidders, max_value, value_pdf)

    # Compute a sharp identified set on the mean and the variance
    print("Doing econometrics on BCE.")
    lb_mean, ub_mean = bce_lp.inverse_bce(
        num_bidders, max_value, bid_pdf, utils.first_moment, tolerance=opts.tolerance)
    true_mean = np.dot(value_pdf, [utils.first_moment(v)
                                   for v in range(max_value + 1)])

    lb_var, ub_var = bce_lp.inverse_bce(
        num_bidders, max_value, bid_pdf, utils.second_moment)
    true_var = np.dot(value_pdf, [utils.second_moment(v)
                                  for v in range(max_value + 1)])

    revenue = sum([max(bids) * bid_pdf[bids] for bids in bid_pdf])
    easy_bound = utils.brooks_bound(num_bidders, max_value, bid_pdf)

    # Print results
    print("\n\nINSTANCE PROPERTIES\n-------------------------")
    print("Common value PDF: {}".format(value_pdf))
    print("BCE distribution of bids: {}".format(bid_pdf))
    print("Revenue of BCE (lower bound on mean): {}".format(revenue))
    print("BBM upper bound on mean: {}".format(easy_bound))

    print("\n\nIDENTIFICATION RESULTS\n------------------------")
    print("Interval on mean: ({}, {}). True value of mean: {}".format(
        lb_mean, ub_mean, true_mean))
    print("Interval on second moment: ({}, {}). True second moment: {}".format(
        lb_var, ub_var, true_var))
    print("Inferred Standard deviation: ({},{}). True standard deviation: {}".format(
        math.sqrt(max([lb_var - ub_mean**2, 0])),
        math.sqrt(ub_var - lb_mean**2),
        math.sqrt(true_var - true_mean**2)))
    print("\n\n")


def main(args):
    """
    Main Function. Example call:

    python single_bce_inference.py --num_bidders 2 --max_value 20 --num_samples 1000 --max_trials 30 --random_seed 12345 --tolerance 0.0001 --value_pdf_type uniform --plot 0

    INPUTS:
    - args: (list of strings) command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Robust identification in auctions! Example call: \
                    python single_bce_inference.py --num_bidders 2 --max_value 20 --num_samples 1000 --max_trials 30 --random_seed 12345 --tolerance 0.0001 --value_pdf_type uniform --plot 0")
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
    parser.add_argument("--plot", dest="plot",
                        type=int, help='Whether to plot figures', default=0)
    opts = parser.parse_args(args[1:])

    mean_and_variance_analysis(opts)


if __name__ == '__main__':
    main(sys.argv)
