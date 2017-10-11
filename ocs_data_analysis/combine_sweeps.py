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

def main(args):
    parser = argparse.ArgumentParser(
        description="Robust identification in auctions!")
    parser.add_argument("--mgr", dest="mu_gr",
                        type=str, help='Grid of mus', default="(0.5,0.5,10)")
    parser.add_argument("--sgr", dest="sigma_gr",
                        type=str, help='Grid of sigma', default="(0.5,0.5,10)")
    parser.add_argument("--sweep_num", dest="sweep_num",
                        type=int, help='Number of sweep parametric tasks in the cluster', default=1)
    parser.add_argument("--out", dest="out_file",
                        type=str, help='Output file for sharp set matrix', default="sharp-set")
    opts = parser.parse_args(args)



    result_list = []
    for sweep_it in range(0, opts.sweep_num):
        sweep_np = np.load("{}_{:03d}.npy".format(opts.out_file, sweep_it+1))
        result_list.extend(sweep_np.tolist())

    min_mu, step_mu, max_mu = eval(opts.mu_gr)
    min_sigma, step_sigma, max_sigma = eval(opts.sigma_gr)
    mu_vect = [
        x * step_mu for x in range(int(min_mu / step_mu), int((max_mu + step_mu) / step_mu))]
    sigma_vect = [x * step_sigma for x in range(
        int(min_sigma / step_sigma), int((max_sigma + step_sigma) / step_sigma))]

    lower_bound_mat = []
    num_results = len(result_list)
    cnt = 0
    for mu in mu_vect:
        lower_bound_mu = []
        for sigma in sigma_vect:
            if cnt < num_results:
                lower_bound_mu.append(result_list[cnt])
            else:
                lower_bound_mu.append(-1)
            cnt += 1
        lower_bound_mat.append(lower_bound_mu)

    for index_mu in range(len(mu_vect)):
        for index_sigma in range(len(sigma_vect)):
            if lower_bound_mat[index_mu][index_sigma] == 'Infeasible':
                lower_bound_mat[index_mu][index_sigma] = -1

    lower_bound_mat = np.asarray(lower_bound_mat)

    np.save(opts.out_file, lower_bound_mat)


if __name__ == '__main__':
    main(sys.argv[1:])
