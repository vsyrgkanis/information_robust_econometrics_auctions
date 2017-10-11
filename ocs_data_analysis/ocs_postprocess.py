#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import argparse
sys.path.append(os.path.abspath('../lib'))
from bcemetricspy import bce_lp, parse_file, utils
import openpyxl
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pylab as plt

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


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
    parser.add_argument("--bid_thr", dest="bid_thr",
                        type=int, help='Maximum Bid Threshold', default=18000)
    parser.add_argument("--non_param", dest="non_param", type=int,
                        help='Whether to run non parametric analysis too. Τoo slow for local computation',
                        default=0)
    parser.add_argument("--in", dest="in_file",
                        type=str, help='Input file for sharp set matrix', default="sharp-set.npy")
    parser.add_argument("--tol", dest="tol",
                        type=float, help='Tolerance used to plot identified sets', default=-1)
    parser.add_argument("--tol_file", dest="tol_file",
                        type=str, help='Tolerance file saved in np array', default=None)

    opts = parser.parse_args(args)

    # Output directory
    output_dir = os.path.dirname(opts.in_file)

    # Get input data
    nb_bidders = opts.nb_bidders
    max_value = opts.max_value
    max_bid = opts.max_bid
    mu_gr = opts.mu_gr
    sigma_gr = opts.sigma_gr

    tol_matrix = np.load(opts.in_file)
    if os.path.isfile(opts.tol_file):
        tol = np.load(opts.tol_file)
    else:
        tol = opts.tol
    # If we don't specify a tolerance, set the default to the minimum tolerance
    if tol == -1:
        tol = np.amin(tol_matrix)
    # Parse the grid
    min_mu, step_mu, max_mu = eval(mu_gr)
    min_sigma, step_sigma, max_sigma = eval(sigma_gr)
    mu_vect = np.arange(min_mu, max_mu + step_mu, step_mu)
    sigma_vect = np.arange(min_sigma, max_sigma + step_sigma, step_sigma)

    # Open the data
    wb = openpyxl.load_workbook('ocs_dataset/Dataset.xlsx')
    sheet1 = wb.get_sheet_by_name('Tract79')
    sheet2 = wb.get_sheet_by_name('Trbid79')
    list_bids = parse_file.matrix_bids(
        wb, sheet1, sheet2, nb_bidders, bid_threshold=opts.bid_thr)
    max_real_bid = max([max(line) for line in list_bids])
    bid_pdf = parse_file.bins_from_matrix(list_bids, max_bid)
    nb_bids = len(list_bids[0])
    print("Num Auctions: {}".format(nb_bids))

    # Compute the non-parametric lower and upper bounds
    if opts.non_param == 1:
        lower_bound, upper_bound = bce_lp.inverse_bce_sparse(
            nb_bidders, max_value, bid_pdf, utils.first_moment, tolerance=tol)

        print("Non-parametric lower bound: {}".format(lower_bound *
                                                      max_real_bid / float(max_bid)))
        print("Non-parametric upper bound: {}".format(upper_bound *
                                                      max_real_bid / float(max_bid)))

    # Compute the parametric lower and upper bounds given a level of tolerance
    lower_bound_par = 10**6
    upper_bound_par = 0
    for index_mu in range(len(mu_vect)):
        for index_sigma in range(len(sigma_vect)):
            mu = mu_vect[index_mu]
            sigma = sigma_vect[index_sigma]
            tol_mu_sigma = tol_matrix[index_mu][index_sigma]
            if tol >= tol_mu_sigma:
                density = [np.exp(-(v - mu)**2 / (2 * sigma**2))
                               for v in range(max_value + 1)]
                s = sum(density)
                density = [density[v] / s for v in range(max_value + 1)]
                mean = sum([v * density[v] for v in range(max_value + 1)])
    
                if mean <= lower_bound_par:
                    lower_bound_par = mean
                if mean >= upper_bound_par:
                    upper_bound_par = mean
    
    print("Parametric lower bound: {}".format(
            lower_bound_par * max_real_bid / float(max_bid)))
    print("Parametric upper bound: {}".format(
            upper_bound_par * max_real_bid / float(max_bid)))

    # Observed revenue (first price auction) and Brooks bound
    observed_revenue = sum([max(key) * bid_pdf[key] for key in bid_pdf.keys()])
    upper_brooks_bound = utils.brooks_bound(nb_bidders, max_value, bid_pdf)
    print("Lower bound via observed revenue: {}".format(
        observed_revenue * max_real_bid / float(max_bid)))
    print("Upper bound via Brooks: {}".format(
        upper_brooks_bound * max_real_bid / float(max_bid)))

    print("Maximum observed bid per acre: {}".format(max_real_bid))

    # Plot the distribution of bids and print mean and standard deviation
    plt.figure()
    bid_dist = [0] * (max_bid + 1)

    for key in bid_pdf.keys():
        for bidder in range(nb_bidders):
            bid = int(key[bidder])
            bid_dist[bid] += bid_pdf[key] / nb_bidders
    x_axis = [max_real_bid * b / float(max_bid) for b in range(max_bid + 1)]
    plt.plot(x_axis, bid_dist)
    plt.xlabel("Bid")
    plt.ylabel("Probability")
    plt.show()
    
    sum_proba = 0
    for key in bid_pdf.keys():
      sum_proba += bid_pdf[key]
    
    mean_bid = sum([bid*bid_dist[bid] for bid in range(max_bid + 1)])
    var_bid = sum([bid**2*bid_dist[bid] for bid in range(max_bid + 1)]) - mean_bid**2
    std_bid = math.sqrt(var_bid)
    median_bid = 0
    
    sum_bid = 0
    for bid in range(0,max_bid):
      if sum_bid < 0.5 and sum_bid+bid_dist[bid + 1] >= 0.5:
        median_bid = bid + 1
      sum_bid += bid_dist[bid + 1]
    
    print "The mean of the distribution of bids is {}".format(mean_bid * max_real_bid / float(max_bid))
    print "The standard deviation of the distribution of bids is {}".format(std_bid * max_real_bid / float(max_bid))
    print "The median is given by {}".format(median_bid * max_real_bid / float(max_bid))
    print "The maximum real bid is {}".format(max_real_bid)    

    figname = "bid_distribution_nb_{}_mb_{}_mv_{}_mbr_{}_sgr_{}_bid_thr_{}.png".format(
        opts.nb_bidders, opts.max_bid, opts.max_value, opts.mu_gr, opts.sigma_gr, opts.bid_thr)
    plt.savefig(os.path.join(output_dir, figname))

    # For two bidders, plot a colormap of the distribution
    if nb_bidders == 2:
        plt.figure()
        joint_bid_dist = np.zeros((max_bid + 1, max_bid + 1))
        for key in bid_pdf.keys():
            joint_bid_dist[int(key[0])][int(key[1])] = bid_pdf[key]
        plt.imshow(joint_bid_dist, extent=(0, max_real_bid,
                                           max_real_bid, 0), interpolation='nearest')
        plt.colorbar()
        plt.xlabel("Bidder 2")
        plt.ylabel("Bidder 1")
        plt.show()

        figname = "joint_bid_distribution_nb_{}_mb_{}_mv_{}_mbr_{}_sgr_{}_bid_thr_{}.png".format(
            opts.nb_bidders, opts.max_bid, opts.max_value, opts.mu_gr, opts.sigma_gr, opts.bid_thr)
        plt.savefig(os.path.join(output_dir, figname))

    # Plot the colormap
    plt.figure()
    min_mu_renorm = min_mu * max_real_bid / float(max_bid)
    max_mu_renorm = max_mu * max_real_bid / float(max_bid)
    min_sigma_renorm = min_sigma * max_real_bid / float(max_bid)
    max_sigma_renorm = max_sigma * max_real_bid / float(max_bid)
    plt.imshow(tol_matrix, extent=(min_sigma_renorm, max_sigma_renorm,
                                   max_mu_renorm, min_mu_renorm), interpolation='nearest')
    plt.colorbar()
    plt.xlabel("$\sigma$")
    plt.ylabel("$\mu$")

    figname = "colormap_nb_{}_mb_{}_mv_{}_mbr_{}_sgr_{}_bid_thr_{}.png".format(
        opts.nb_bidders, opts.max_bid, opts.max_value, opts.mu_gr, opts.sigma_gr, opts.bid_thr)
    plt.savefig(os.path.join(output_dir, figname))

    # Plot the sharp identified set
    min_tol = np.amin(tol_matrix)
    overlap_matrix = np.zeros((len(mu_vect), len(sigma_vect)))
    
    for index_mu in range(len(mu_vect)):
        for index_sigma in range(len(sigma_vect)):
            if tol_matrix[index_mu, index_sigma] <= min_tol and tol_matrix[index_mu, index_sigma] <= tol:
                overlap_matrix[index_mu, index_sigma] = 1
            #1 for an element both in the min tol set and the quantile set
            if tol_matrix[index_mu, index_sigma] > min_tol and tol_matrix[index_mu, index_sigma] <= tol:
                overlap_matrix[index_mu, index_sigma] = 0.5
                #0.5 for an element in the quantile set but not in the min tol set
               
    
    plt.figure()
    min_mu_renorm = min_mu * max_real_bid / float(max_bid)
    max_mu_renorm = max_mu * max_real_bid / float(max_bid)
    min_sigma_renorm = min_sigma * max_real_bid / float(max_bid)
    max_sigma_renorm = max_sigma * max_real_bid / float(max_bid)
    plt.imshow(overlap_matrix,
               extent=(min_sigma_renorm, max_sigma_renorm,
                       max_mu_renorm, min_mu_renorm),
               interpolation='nearest')
    plt.colorbar()
    plt.xlabel("$\sigma$")
    plt.ylabel("$\mu$")

    figname = "sharp_set_nb_{}_mb_{}_mv_{}_mbr_{}_sgr_{}_bid_thr_{}.png".format(
        opts.nb_bidders, opts.max_bid, opts.max_value, opts.mu_gr, opts.sigma_gr, opts.bid_thr)
    plt.savefig(os.path.join(output_dir, figname))
    
    # Plot the sharp identified set in the (mean,standard deviation) space
    
    tol_meanvar_matrix = np.zeros((len(mu_vect), len(sigma_vect)));
    for index_mu in range(len(mu_vect)):
      for index_sigma in range(len(sigma_vect)):
        mu = mu_vect[index_mu]
        sigma = sigma_vect[index_sigma]
        if tol_matrix[index_mu, index_sigma] <= min_tol or tol_matrix[index_mu, index_sigma] <= tol:
          density = [np.exp(-(v - mu)**2 / (2 * sigma**2))
                     for v in range(max_value + 1)]
          s = sum(density)
          density = [density[v] / s for v in range(max_value + 1)]
          mean = sum([v*density[v] for v in range(max_value + 1)])
          var = sum([v**2*density[v] for v in range(max_value + 1)]) - mean**2
          std = math.sqrt(var)
          
          rounded_mean = find_nearest(mu_vect,mean)
          rounded_std = find_nearest(sigma_vect,std)
          
          index_mean = np.where(mu_vect==rounded_mean)
          index_std = np.where(sigma_vect==rounded_std)
          
          if tol_matrix[index_mu, index_sigma] <= min_tol and tol_matrix[index_mu, index_sigma] <= tol:
            tol_meanvar_matrix[index_mean,index_std]  = 1
            #1 for an element both in the min tol set and the quantile set
          if tol_matrix[index_mu, index_sigma] > min_tol and tol_matrix[index_mu, index_sigma] <= tol and tol_meanvar_matrix[index_mean,index_std] != 1:
          #the last check is so that if two values of (mu,sigma) map to the same (mean,std) the tolerance set, one is in the min tolerance set and the other not, the corresponding (mean,std) is in the min tolerance set
            tol_meanvar_matrix[index_mean,index_std]  = 0.5
            #0.5 for an element in the quantile set but not in the min tol set
          
    plt.figure()
    min_mu_renorm = min_mu * max_real_bid / float(max_bid)
    max_mu_renorm = max_mu * max_real_bid / float(max_bid)
    min_sigma_renorm = min_sigma * max_real_bid / float(max_bid)
    max_sigma_renorm = max_sigma * max_real_bid / float(max_bid)
    plt.imshow(tol_meanvar_matrix,
               extent=(min_sigma_renorm, max_sigma_renorm,
                       max_mu_renorm, min_mu_renorm),
               interpolation='nearest')
    plt.colorbar()
    plt.xlabel("Standard deviation")
    plt.ylabel("Mean")

    figname = "sharp_set_meanstd_nb_{}_mb_{}_mv_{}_mbr_{}_sgr_{}_bid_thr_{}.png".format(
        opts.nb_bidders, opts.max_bid, opts.max_value, opts.mu_gr, opts.sigma_gr, opts.bid_thr)
    plt.savefig(os.path.join(output_dir, figname))
    

if __name__ == '__main__':
    main(sys.argv[1:])
