## This file takes as an input two matrices:
## - a 2D matrix in_tol(mu,sigma) of minimum tolerances on the data 
## for each (mu,sigma) pair
## - a 3D matrix in_subsample(subsample,mu,sigma) of minimum tolerances for each 
## subsample and each (mu,sigma) pair
## The file returns an estimation of the qtl_prcent quantile of the maximum over
## (mu,sigma) in the identified set of the minimum tolerance for a given
## (mu,sigma) pair  



import numpy as np
import argparse
from scipy.stats import norm
import sys

#Takes a vector of samples from a distribution and estimates the qtl_percent
#quantile of said distribution
def qtl_estimation(sample_vect,qtl_percent):
    qtl = np.percentile(sample_vect,100*qtl_percent)          
    return qtl

#Returns the vector of the supremum over (mu,sigma) in the minimum tolerance set
#of the minimum tolerance for each subsample. 
def supmin_tol_samples(in_min_tol,in_subsamples,dist):
    min_tol = np.amin(in_min_tol)
    if dist == "gaussian":
        (nb_samples,size_mu,size_sigma) = in_subsamples.shape
        supmin_tol_vect = np.zeros((nb_samples,))
        for index_sample in range(nb_samples):
            #We determine the supremum--over (mu,sigma) in the minimum tolerance 
            #set--of the minimum tolerance for a given (mu,sigma) in subsample 
            #index_sample
            supmin = 0
            for index_mu in range(size_mu):
                for index_sigma in range(size_sigma):
                    if in_min_tol[index_mu,index_sigma] <= min_tol:
                        if supmin <= in_subsamples[index_sample,index_mu,index_sigma]:
                            supmin = in_subsamples[index_sample,index_mu,index_sigma]
            supmin_tol_vect[index_sample] = supmin
        return supmin_tol_vect
    if dist == "poisson" or dist == "geometric" or dist == "binomial":
        (nb_samples,size_param) = in_subsamples.shape
        supmin_tol_vect = np.zeros((nb_samples,))
        for index_sample in range(nb_samples):
            #We determine the supremum--over the parameters in the minimum tolerance 
            #set--of the minimum tolerance for a given parameter value in subsample 
            #index_sample
            supmin = 0
            for index_param in range(size_param):
                if in_min_tol[index_param] <= min_tol:
                    if supmin <= in_subsamples[index_sample,index_param]:
                        supmin = in_subsamples[index_sample,index_param]
            supmin_tol_vect[index_sample] = supmin
        return supmin_tol_vect
    
def unit_tests():
    #For the testing phase, I use 10**4 samples because I want to observe 
    #convergence to the true parameters/cdfs
    nb_samples = 10**4
    quantile_percent = 0.95
    
    #Generate a small grid of (mu,sigma):
    mu_gr = "(0.5,0.5,10)"
    sigma_gr = "(0.5,0.5,10)"
    min_mu, step_mu, max_mu = eval(mu_gr)
    min_sigma, step_sigma, max_sigma = eval(sigma_gr)
    mu_vect = np.arange(min_mu, max_mu + step_mu, step_mu)
    sigma_vect = np.arange(min_sigma, max_sigma + step_sigma, step_sigma)
    size_mu=len(mu_vect)
    size_sigma=len(sigma_vect)
    
    #Generate a fake input containing nb_samples samples from a normal distribution
    #with mean mu and variance sigma, for every (mu,sigma) pair in the grid
    np.random.seed(seed=1561)
    test_matrix_std = np.zeros((nb_samples,size_mu,size_sigma))

    for index_sample in range(nb_samples):
        for index_mu in range(size_mu):
            for index_sigma in range(size_sigma):
                mu = mu_vect[index_mu]
                sigma = sigma_vect[index_sigma]
                test_matrix_std[index_sample,index_mu,index_sigma] = np.random.normal(mu,sigma)
    
    print("Starting testing phase...")
    
    #Testing the quantile function using samples from a standard Gaussian
    quantile_percent = 0.95
    test_vect_qtl = np.zeros((nb_samples,))
    for index_sample in range(nb_samples):
        test_vect_qtl[index_sample] = np.random.normal(0,1)
    quantile = qtl_estimation(test_vect_qtl, quantile_percent)
    rv = norm()
    cdf_quantile = rv.cdf(quantile)
    
    if abs(cdf_quantile - quantile_percent) < 0.01:
        print("Quantile test passed!")
    else:
        print("Quantile test failed...")
        
    ## Unit test using simple parameters:
    nb_samples = 100
    quantile_percent = 0.90

    #2D matrix M(mu,sigma) of minimum tolerances on the data for each (mu,sigma) 
    #pair
    input_min_tol = np.load("min_tolerance.npy")
    #3D matrix M(subsample,mu,sigma) of minimum tolerances for each subsample
    #and each (mu,sigma) pair
    input_subsamples = np.load("sample_min_tolerance.npy")
    
    ##Compute the quantile-based tolerance
    supmin_tol_vect = supmin_tol_samples(input_min_tol,input_subsamples,"gaussian")
    qtl_min_tol = qtl_estimation(supmin_tol_vect,quantile_percent)

    if qtl_min_tol == 2:
        print("Default parameters test passed!")
    else:
        print("Default parameters test failed...")
        
    print("Testing phase done!")
    
    

def main(args):
    parser = argparse.ArgumentParser(
        description="Quantile estimation!")
    parser.add_argument("--qtl", dest="qtl_prct",
                        type=float, help='Desired percentile', default=0.90)
    parser.add_argument("--in_sub", dest="in_subsamples",
                        type=str, help='Prefix of input file of subsampled min tolerances', default="./output/subsamples")
    parser.add_argument("--in_tol", dest="in_tol",
                        type=str, help='Input file of min tolerances', default="./output/spl_tolerance_matrix_nbbids=500")
    parser.add_argument("--out", dest="out_file",
                        type=str, help='Output file containing the minimum tolerance', default="./output/quantile_tol")
    parser.add_argument("--dist", dest="dist",
                        type=str, help='Distribution we parameterize on', default="gaussian")
    
    opts = parser.parse_args(args)
   
    ## Get input data and parse the grid
    quantile_percent = opts.qtl_prct
    dist = opts.dist

    #2D matrix M(mu,sigma) of minimum tolerances on the data for each (mu,sigma) 
    #pair
    filename = opts.in_tol
    input_min_tol = np.load(filename)
    #3D matrix M(subsample,mu,sigma) of minimum tolerances for each subsample
    #and each (mu,sigma) pair
    filename = opts.in_subsamples
    input_subsamples = np.load(filename)
    
    ##Compute the quantile-based tolerance
    supmin_tol_vect = supmin_tol_samples(input_min_tol,input_subsamples,dist)
    qtl_min_tol = qtl_estimation(supmin_tol_vect,quantile_percent)
    
    print(str(100*quantile_percent)+"% quantile for {}: {}".format(dist, qtl_min_tol))
    
    filename = opts.out_file
    np.save(filename,qtl_min_tol)

if __name__ == '__main__':
    ## We start with a testing phase to determine whether the functions work!
    #print("Testing phase...")
    #unit_tests()
    
    
    print("Determining the tolerances from the data...")
    main(sys.argv[1:])
    
