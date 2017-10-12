# Python code for doing robust econometrics in auctions


## Contents
* ocs_data_analysis: code for analyzing the ocs dataset. also contains the dataset itself.
 
* lib/bcemetrics
  - lib/bcemetrics/bce_lp.py - This code contains the main functions that build LPs related to Bayes-Correlated
    equilibrium. This is supposed to be a library code that can be called for whatever we want
    to do. It contains functions for computing a BCE and for inverting a BCE
  - lib/bcemetrics/parse_file.py - This code contains utility functions for reading in the OCS data set into a
    a data structure that the rest of the code can use to run. It returns a representation of the BCE in the OCS
    data
  - lib/bcemetrics/utils.py - Some handy util functions

* Monte Carlo Simulations (aka simulated data analysis)
  * example_inference.ipynb - Jupyter notebook explaining how to interface with the library and applying it to 
    some toy monte carlo data.
  * many_bce_inference.py - For a single valuation distribution, this code computes many BCE and then 
    runs inference on each one of them to compute the sharp set for the mean of the distribution. 
    Then it stores the statistics of the sharp set for each of these BCE and potentially also plots them.
  * single_bce_inference.py - Computes the sharp identified set for a single bce, for both the mean and the variance

* Parametric analysis of OCS data
    * ocs_inference.sh: This bash script runs all the pipeline for analyzing the ocs data and creating confidence
      sets of the identified set of parameters via sub-sampling
    * Python files
      * parallel_common_value_ocs_sparse.py - This code runs the inverse parametric sparse BCE LP for each possible
        parameter in a discretized grid of the parameter space and returns the mininum tolerance that makes this parameter
        a feasible observed BCE. In fact each call to this file will cover a subset of the parameters dependent on which
        sweep of the parametric cluster job it is run by. 
      * combine_sweeps.py - combines all the sweep files produced by the parallel_common_value_ocs.py run on different
        parts of the parameter space. Then it combines them into a single .npy file with the tolerance matrix for each 
        possible parameter. 
      * ocs_postprocess.py - given a minimum tolerance matrix for each parameter in the discrete grid, this performs
        a post processing analysis of the results. 
      * combine_sub_sweeps.py - this is a helper python code that can is used when the big cluster job fails and then 
        we need to run the code for a sub-part of the parameter space. Then this code joins the sub_sweeps into a file 
        that can be used by combine_sweeps.py

## Usage instructions
* Install python.
  - For linux/Mac it should already be there. 
  - For windows it's easier to install the Anaconda package, which installs almost
  all the above python libs (i.e. numpy, pandas, scipy, matplotlib). https://www.continuum.io/downloads

* Python libraries that are needed (in case you don't already have them). Anaconda 
  already installs most of them.
  - numpy: sudo pip install numpy
  - pandas: sudo pip install pandas
  - scipy: sudo pip install scipy
  - matplotlib: sudo pip install matplotlib

* To run any of these functions you need to install the COIN-CBC linear programming solver.
  - For linux you just run: 
	$ sudo apt-get install coinor-cbc
  - For Mac OS X:
  $ svn co https://projects.coin-or.org/svn/Cbc/stable/2.9 Cbc-2.9
  $ cd Cbc-2.9
  $ make
  $ make install
  Then you need to add the bin path that is created inside this folder to your bash path. You
  can do this by:
  $ sudo vim .bash_profile
  Then adding a line at the end:
  export PATH="__path_to_cbc_bin__:$PATH"
For more installation instructions check here: https://projects.coin-or.org/Cbc

* You also need to install the pulp library for python. Because pulp is constantly
  updating and has some bugs. I recommend downloading the git repo and running setup.py
  - To download the git repo run:
  	$ git clone https://github.com/coin-or/pulp.git
  - Then inside the folder created by git, run:
	- for windows in a power shell
		$ python setup.py install 
    	- for linux/Mac in a terminal:
		$ sudo python setup.py install

* Now you should be able to run the python code.	

## Markdown and Python style

* Should follow pylint guidelines.
* The Markdown in this project should follow the style guide at
  https://github.com/carwin/markdown-styleguide
* The Python could follow the style guide at
  https://google.github.io/styleguide/pyguide.html
