"""
@authors: Vasilis Syrgkanis, Juba Ziani
@date: 4/11/2017

"""
import openpyxl
import numpy as np
# import random
import sys
import os
import argparse
from itertools import product
import time
import multiprocessing
from joblib import Parallel, delayed
sys.path.append(os.path.abspath('../lib'))
from bcemetricspy import bce_lp, parse_file, utils
import random

def run_parametric_lp(mu, sigma, max_value, nb_bidders, bid_pdf):

    density = [np.exp(-(v - mu)**2 / (2 * sigma**2))
               for v in range(max_value + 1)]
    s = sum(density)
    density = [density[v] / s for v in range(max_value + 1)]

    min_tolerance = bce_lp.inverse_bce_parameterized_sparse_min_tolerance(
        nb_bidders, max_value, bid_pdf, utils.first_moment, density, solver_str='COIN')

    return min_tolerance


def main(args):
    parser = argparse.ArgumentParser(
        description="Robust identification in auctions!")
    parser.add_argument("--nb", dest="nb_bidders",
                        type=int, help='Number of bidders', default=2)
    parser.add_argument("--mb", dest="max_bid",
                        type=int, help='Maximum bid', default=10)
    parser.add_argument("--mv", dest="max_value",
                        type=int, help='Maximum value', default=20)
    parser.add_argument("--mgr", dest="mu_gr",
                        type=str, help='Grid of mus', default="(0.5,0.5,10)")
    parser.add_argument("--sgr", dest="sigma_gr",
                        type=str, help='Grid of sigma', default="(0.5,0.5,10)")
    parser.add_argument("--sweep_start_id", dest="sweep_start_id",
                        type=int, help='From which parameter index is this sweep starting', default=0)
    parser.add_argument("--sweep_end_id", dest="sweep_end_id",
                        type=int, help='Until which parameter index is this sweep going', default=-1)
    parser.add_argument("--sweep_it", dest="sweep_it",
                        type=int, help='Sweep parameter iteration in the cluster, ranges from 1 to sweep_num', default=1)
    parser.add_argument("--sweep_num", dest="sweep_num",
                        type=int, help='Number of sweep parametric tasks in the cluster', default=1)
    parser.add_argument("--out_pre", dest="out_pre",
                        type=str, help='Output prefix for tolerance matrix for each subsample', default="/output/subsamples")
    parser.add_argument("--bid_thr", dest="bid_thr",
                        type=int, help='Maximum Bid Threshold', default=100000)
    parser.add_argument("--nb_sples", dest="nb_sples",
                        type=int, help='Number of subsamples', default=100)
    parser.add_argument("--sple_size", dest="sple_size",
                        type=int, help='Size of a subsample', default=50)
    opts = parser.parse_args(args)

    nb_bidders = opts.nb_bidders
    nb_subsamples = opts.nb_sples
    sample_size = opts.sple_size

    print("Sweep it: {} out of {}".format(opts.sweep_it, opts.sweep_num))

    # Open the data
    wb = openpyxl.load_workbook('ocs_dataset/Dataset.xlsx')
    sheet1 = wb.get_sheet_by_name('Tract79')
    sheet2 = wb.get_sheet_by_name('Trbid79')

    list_bids = parse_file.matrix_bids(
        wb, sheet1, sheet2, nb_bidders, bid_threshold=opts.bid_thr)
    nb_bids = len(list_bids[0])
    print("Num Auctions: {}".format(nb_bids))

    # Parameters:
    # renormalize the valuation space to be between 0 and max_value
    max_value = opts.max_value
    max_bid = opts.max_bid  # asse the bids we have observed go from 0 to max_bid

    # Renormalize the bids to be between 0 and max_bid
    max_real_bid = max([max(line) for line in list_bids])
    size_bin = max_real_bid / max_bid
    for index in range(nb_bids):
        for bidder in range(nb_bidders):
            list_bids[bidder][index] = list_bids[
                bidder][index] / float(size_bin)

    # Compute the joint distribution of bids for players 1 and 2.
    # Round values to the nearest integer bin to decide on a bin
    bid_pdf = {}
    for index in range(nb_bids):
        tple = [round(list_bids[bidder][index])
                for bidder in range(nb_bidders)]
        tple = tuple(tple)
        bid_pdf[tple] = 0

    for index in range(nb_bids):
        tple = [round(list_bids[bidder][index])
                for bidder in range(nb_bidders)]
        tple = tuple(tple)
        bid_pdf[tple] += 1 / float(nb_bids)

    # Run the modified LP with the constraint that the density has to be a truncated
    # Gaussian with given mean and variance
    min_mu, step_mu, max_mu = eval(opts.mu_gr)
    min_sigma, step_sigma, max_sigma = eval(opts.sigma_gr)
    mu_vect = np.arange(min_mu, max_mu + step_mu, step_mu)
    sigma_vect = np.arange(min_sigma, max_sigma + step_sigma, step_sigma)

    print("Solving the parametrized LPs...")

    num_cores = multiprocessing.cpu_count()
    print("Num Cores:{}".format(num_cores))
    num_workers = num_cores
    t = time.time()

    param_list = [(mu, sigma)
                  for (mu, sigma) in product(*[mu_vect, sigma_vect])]
    num_params = len(param_list)
    # Selecting the subset of params for this sweep
    if opts.sweep_end_id == -1:
        opts.sweep_end_id = num_params
    param_list = param_list[opts.sweep_start_id:opts.sweep_end_id]
    num_params = len(param_list)
    # Compute the parameters for the specific sweep tasks
    # Split range of parameters to sweep_num approximately equally sized
    # batches
    sweep_batches = np.array_split(range(0, num_params), opts.sweep_num)
    # We are now running the sweep_it-1 indexed batch
    sweep_batch = sweep_batches[opts.sweep_it - 1]
    sweep_start = sweep_batch[0]
    sweep_end = sweep_batch[-1] + 1
    print("Running for range: {}:{}".format(sweep_start, sweep_end))
    sweep_params = param_list[sweep_start:sweep_end]

    # Create a list of bids to subsample from
    list_bids = []
    for key in bid_pdf:
        nb_occurences_bid = int(round(bid_pdf[key]*nb_bids))
        for count in range(nb_occurences_bid):
            list_bids.append(key)


    # Run the core of the code
    random.seed(2563)
    for subsample in range(nb_subsamples):
        # Subsample auctions and obtain bid_pdf_subsample
        random.shuffle(list_bids)
        bid_pdf_subsample = {}
        for index in range(sample_size):
            bids = list_bids[index]
            bid_pdf_subsample[bids] = 0

        for index in range(sample_size):
            bids = list_bids[index]
            bid_pdf_subsample[bids] += 1 / float(sample_size)

        # Run the min tolerance code in parallel
        result_list = Parallel(n_jobs=num_workers, verbose=6)(delayed(run_parametric_lp)(
                    mu, sigma, max_value, nb_bidders, bid_pdf_subsample) for (mu, sigma) in sweep_params)

        #Save the results
        filename = opts.out_pre+'_sple_'+str(subsample)+'.npy'
        print("Output file: {}".format(filename))
        np.save(filename, np.asarray(result_list))

    print('Total time to run all params: {}s'.format(time.time() - t))



if __name__ == '__main__':
    main(sys.argv[1:])
