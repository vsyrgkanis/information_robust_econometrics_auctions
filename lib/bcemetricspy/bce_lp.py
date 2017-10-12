"""
@author: Vasilis Syrgkanis
@date: 2/24/2017

Compute a BCE of a common value first price auction and compute the sharp
identified set of any moment given a bid distribution of a BCE.
"""
from itertools import product
import random
import numpy as np
import pulp as plp
import openpyxl


def deviation_term(dev_bid, bidder_id, value, bid_vector, psi_var, prob):
    """ Returns an LP term of the form q*psi(v,b)*(deviating_utility-current_utility)
    where psi(v,b) is the LP variable which is either Pr[v | b] or Pr[b | v]
    depending on whether we are in the BCE LP or the Inverse BCE LP and
    q = Pr[v] or P[b] depending on which LP it is.

    Parameters
    ----------
    - dev_bid : (int) deviating bid_vector
    - bidder_id: (int) id of bidder who is deviating
    - value : (int) the common value
    - bid_vector : (tuple of ints) the current bid vector from which deviation is computed
    - psi_var : (pulp.variable) the LP variable associated with value and bid_vector
    - prob : (float) the input marginal to the LP, either Pr[v] or Pr[b]
    """

    # Separate the bid of the bidder we are looking at from the opponent bids
    cur_bid = bid_vector[bidder_id]
    opponent_bids = bid_vector[0:bidder_id] + bid_vector[bidder_id + 1:]
    # Compute the maximum other bid
    max_other_bid = max(opponent_bids)

    # The allocation probability in the current bid profile. Random
    # tie-breaking
    cur_allocation = 0 if cur_bid < max_other_bid else (
        1 if cur_bid > max_other_bid else 1 / (1. + opponent_bids.count(cur_bid)))
    # Current utility is current_allocation * (value - current_bid)
    cur_utility = cur_allocation * (value - cur_bid)
    # Allocation probability if player deviated to dev_bid
    dev_allocation = 0 if dev_bid < max_other_bid else (
        1 if dev_bid > max_other_bid else 1 / (1. + opponent_bids.count(dev_bid)))
    # Deviating utility
    dev_utility = dev_allocation * (value - dev_bid)

    # Compute the LP term of the form psi(v,b)*(deviating_utility-current_utility)
    # If both utilities are the same, then we return 0 to avoid the computational
    # overhead of symbolic multiplication from the pulp library.
    if dev_utility == cur_utility:
        return 0
    else:
        coefficient = (dev_utility - cur_utility) * prob
        return coefficient * psi_var


def get_solver(solver_str):
    if solver_str == 'GLPK':
        return plp.solvers.GLPK(mip=0, msg=0)
    elif solver_str == 'GUROBI':
        return plp.solvers.GUROBI(mip=False, msg=False)
    else:
        return plp.solvers.COIN_CMD(mip=0, msg=0)


def compute_bce(num_bidders, max_value, value_pdf, num_samples=100, max_trials=20, random_seed=1232, solver_str='COIN'):
    """ Computes a BCE with a random objective. Returns the marginal distribution
    of bid vectors that is observed from this BCE. It samples a set of num_samples
    potential bid vectors and tries to compute a BCE supported on this bid vectors.
    For a random coefficient vector c and a random sample of bid vectors S, it
    constructs the following BCE-LP:

    max_{Pr[b | v]} sum_{b in S} Pr[b | v] * c(b) \n
    forall i, b_i*, b_i': \n
     sum_{b in S: b_i=b_i*, v in {0,...,max_value}} \n
                Pr[b | v] * pi(v) * (U_i(b_i',b_{-i};v)-U_i(b;v)) <= 0 \n
     forall v: sum_{b in S} Pr[b | v] = 1 \n

    Parameters
    ----------
    - num_bidders : (int) the number of bidders
    - max_value : (int) the maximum value/bid. Values and bids are constraint
        in {0,1,...,max_value}
    - value_pdf : (list of floats) the pdf pi of the value distribution
        as a list of length max_value+1
    - num_samples : (int) the number of bid vectors to sample as a potential support of the BCE
    - max_trials : (int) how many times to try to draw samples of bid vectors and try to find
        BCE supported on this bid vectors
    - random_seed : (int) a random seed for the bid vector sampling generation and the objective
        generation

    Returns
    -------
    - bid_pdf : (dict of tuple -> double) the marginal bid vector distribution of the compute
      BCE. The keys are a tuple of the bid vector and the value is the pdf of the tuple. It only
      contains the tuples with positive mass.
    """

    status = 0  # whether the LP was feasible
    trials = 0  # counter on sampling trials
    random.seed(random_seed)  # seed the random generator
    while status != 1 and trials < max_trials:  # while we have not found a feasible LP

        prob = plp.LpProblem("BCE", plp.LpMinimize)  # create an LP instance

        # Sample a set of bid vectors to try to create a BCE with just them
        bid_vectors = product(range(max_value + 1), repeat=num_bidders)
        pool = tuple(bid_vectors)
        indices = random.sample(range(len(pool)), min(num_samples, len(pool)))
        sampled_bid_vectors = set([pool[i] for i in indices])

        # Variables are of the form (v,b) where v is a value in {0,...,max_value}
        # and b is a bid vector in the sampled set
        lp_var_keys = product(*[range(max_value + 1), sampled_bid_vectors])

        # Create the psi variables which correspond to Pr[b | v] for each (v,b)
        # pair
        psi_vars = plp.LpVariable.dicts('psi', lp_var_keys, lowBound=0)

        # Creating the best response constraints
        devs = product(
            *[range(max_value + 1), range(max_value + 1), range(0, num_bidders)])
        for (cur_bid, dev_bid, bidder_id) in devs:  # for all b_i*, b_i', i
            if cur_bid != dev_bid:  # if b_i* \neq b_i'
                # Create all the terms of the form Pr[b | v] * pi(v) * (U_i' - U_i)
                # for all b \in S, such that b_i = b_i*
                dev_terms = [
                    deviation_term(dev_bid, bidder_id, v, bids,
                                   psi_vars[(v, bids)], value_pdf[v])
                    for (v, bids) in product(*[range(max_value + 1), sampled_bid_vectors])
                    if bids[bidder_id] == cur_bid]
                # Add these terms to create the best response constraint
                prob += plp.lpSum(dev_terms) <= 0, "Dev_{}_{}_{}".format(
                    cur_bid, dev_bid, bidder_id)

        # Constraint that Pr[b | v] is a distribution for each fixed v
        for value in range(max_value + 1):
            cond_vars = [psi_vars[(value, bids)]
                         for bids in sampled_bid_vectors]
            prob += plp.lpSum(cond_vars) == 1, "Density_Psi_{}".format(value)

        # Objective coefficients are random numbers based on the seed
        np.random.seed(random_seed)
        prob += plp.lpSum([np.random.standard_normal(1) *
                           var for var in psi_vars.values()])

        # Solve LP
        prob.solve(get_solver(solver_str))

        # Get the status returned by the solver. 1 means success
        status = int(prob.status)

        # Increase the trial counter
        trials += 1

    if status == 1:
        # Once we have found a BCE, compute the marginal bid vector
        # distribution
        bid_pdf = {}
        for bid_vector in sampled_bid_vectors:
            # Compute the probability of the bid vector: sum_{v} Pr[b | v] *
            # pi(v)
            prob_mass = sum([plp.value(psi_vars[(v, bid_vector)]) * value_pdf[v]
                             for v in range(max_value + 1)])
            # If mass is positive add it to the bid_pdf dictionary
            if prob_mass > 0:
                bid_pdf[bid_vector] = prob_mass

        return bid_pdf
    else:
        return None


def inverse_bce(num_bidders, max_value, bid_pdf, moment_fn, tolerance=0.0001, solver_str='COIN'):
    """Computes a sharp identified set of a moment of the common value
    distribution from an observed bid vector distribution. It constructs
    the following inverse BCE-LP:

    max_{Pr[v | b]} sum_{v in {0,...,max_value}, b in S} f(v) * Pr[v | b] * phi(b) \n
    forall i, b_i*, b_i': \n
    sum_{b in S: b_i=b_i*, v in {0,...,max_value}} \n
                Pr[v | b] * phi(b) * (U_i(b_i',b_{-i};v)-U_i(b;v)) <= tolerance \n
    forall v: sum_{v in {0,...,max_value}} Pr[v | b] = 1 \n

    Parameters
    ----------
    - num_bidders : (int) the number of bidders
    - max_value : (int) the maximum value/bid. Values and bids are constraint in {0,1,...,max_value}
    - bid_pdf : (dictionary from tuple of ints to float) the pdf phi of the bid vector
        distribution as a dictionary with keys the tuple of bids and value the pdf of
        that tuple of bids
    - moment_fn : (function from float to float) the moment function
    - tolerance : (float) a tolerance on the best response constraints in case of sampling
        or numerical error in the bid_pdf

    Returns
    -------
    - (lower_bound, upper_bound) : (double, double) the upper and lower bound of the
      sharp identified set of the moment
    """

    # create an LP instance
    prob = plp.LpProblem("Inverse-BCE", plp.LpMinimize)

    # Create the psi variables which correspond to Pr[v | b] for each (v,b)
    # pair, where b is in the support of the bid_pdf
    lp_var_keys = product(*[range(max_value + 1), bid_pdf.keys()])
    psi_vars = plp.LpVariable.dicts('psi', lp_var_keys, lowBound=0)

    # Creating the best response constraints with tolerance
    devs = product(
        *[range(max_value + 1), range(max_value + 1), range(0, num_bidders)])
    for (cur_bid, dev_bid, bidder_id) in devs:  # for all b_i*, b_i', i
        if cur_bid != dev_bid:  # if b_i* \neq b_i'
            # Create all the terms of the form Pr[v | b] * phi(b) * (U_i' - U_i)
            # for all b \in S, such that b_i = b_i*
            dev_terms = [
                deviation_term(dev_bid, bidder_id, v, bids,
                               psi_vars[(v, bids)], bid_pdf[bids])
                for (v, bids) in product(*[range(max_value + 1), bid_pdf.keys()])
                if bids[bidder_id] == cur_bid]
            # Add these terms to create the best response constraint with
            # tolerance
            prob += plp.lpSum(dev_terms) <= tolerance, "Dev_{}_{}_{}".format(
                cur_bid, dev_bid, bidder_id)

    # Constraint that Pr[v | b] is a distribution for each fixed b
    for bids in bid_pdf.keys():
        cond_vars = [psi_vars[(value, bids)] for value in range(max_value + 1)]
        prob += plp.lpSum(cond_vars) == 1, "Density_Psi_{}".format(bids)

    # Objective coefficients based on moment function for the lower bound
    # Minimize sum_{v, b} f(v) * Pr[v | b] * phi(b)
    prob += plp.lpSum([moment_fn(v) * bid_pdf[bids] * psi_vars[(v, bids)]
                       for (v, bids) in product(*[range(max_value + 1), bid_pdf.keys()])])

    # Solve the LP
    prob.solve(get_solver(solver_str))
    # Get the objective value of the LP which is a lower bound on the moment
    if int(prob.status) == 1:
        lower_bound = plp.value(prob.objective)
    else:
        lower_bound = None

    # Now we compute the upper bound on the moment, i.e.
    # Minimize - sum_{v,b} f(v) * Pr[v | b] * phi(b)
    prob += plp.lpSum([-moment_fn(v) * bid_pdf[bids] * psi_vars[(v, bids)]
                       for (v, bids) in product(*[range(max_value + 1), bid_pdf.keys()])])

    # Solve the LP
    prob.solve(get_solver(solver_str))
    # Get minus the objective value of the LP which is an upper bound on the
    # moment
    if int(prob.status) == 1:
        upper_bound = -plp.value(prob.objective)
    else:
        upper_bound = None

    return (lower_bound, upper_bound)


def inverse_bce_sparse(num_bidders, max_value, bid_pdf, moment_fn, tolerance=0.0001, solver_str='COIN'):
    """Computes a sharp identified set of a moment of the common value
    distribution from an observed bid vector distribution. This method takes advantage
    of sparsity in the support of the BCE, by only creating deviating bids that
    are undominated, given the BCE distribution.

    It constructs the following inverse BCE-LP:

    max_{Pr[v | b]} sum_{v in {0,...,max_value}, b in S} f(v) * Pr[v | b] * phi(b) \n
    forall i, b_i*, b_i' that are undominated: \n
    sum_{b in S: b_i=b_i*, v in {0,...,max_value}} \n
                Pr[v | b] * phi(b) * (U_i(b_i',b_{-i};v)-U_i(b;v)) <= tolerance \n
    forall v: sum_{v in {0,...,max_value}} Pr[v | b] = 1 \n

    Parameters
    ----------
    - num_bidders : (int) the number of bidders
    - max_value : (int) the maximum value/bid. Values and bids are constraint in {0,1,...,max_value}
    - bid_pdf : (dictionary from tuple of ints to float) the pdf phi of the bid vector
        distribution as a dictionary with keys the tuple of bids and value the pdf of
        that tuple of bids
    - moment_fn : (function from float to float) the moment function
    - tolerance : (float) a tolerance on the best response constraints in case of sampling
        or numerical error in the bid_pdf

    Returns
    -------
    - (lower_bound, upper_bound) : (double, double) the upper and lower bound of the
      sharp identified set of the moment
    """

    # create an LP instance
    prob = plp.LpProblem("Inverse-BCE", plp.LpMinimize)

    # Create the psi variables which correspond to Pr[v | b] for each (v,b)
    # pair, where b is in the support of the bid_pdf
    lp_var_keys = product(*[range(max_value + 1), bid_pdf.keys()])
    psi_vars = plp.LpVariable.dicts('psi', lp_var_keys, lowBound=0)

    # Creating the best response constraints with tolerance
    for bidder_id in range(num_bidders):
        # Find what are all the bids the player ever submits under BCE
        cur_bid_support = set([b_vec[bidder_id] for b_vec in bid_pdf.keys()])
        for cur_bid in cur_bid_support:
            # Find the possible values that the maximum other bid can take
            max_other_bid_support = set(
                [max(b_vec[0:bidder_id] + b_vec[bidder_id + 1:]) for b_vec in bid_pdf.keys()
                 if b_vec[bidder_id] == cur_bid])
            dev_bid_support = max_other_bid_support | set(
                [b + 1 for b in max_other_bid_support]) | set([0])
            for dev_bid in dev_bid_support:
                if cur_bid != dev_bid:  # if b_i* \neq b_i'
                    # Create all the terms of the form Pr[v | b] * phi(b) * (U_i' - U_i)
                    # for all b \in S, such that b_i = b_i*
                    dev_terms = [
                        deviation_term(dev_bid, bidder_id, v, bids,
                                       psi_vars[(v, bids)], bid_pdf[bids])
                        for (v, bids) in product(*[range(max_value + 1), bid_pdf.keys()])
                        if bids[bidder_id] == cur_bid]
                    # Add these terms to create the best response constraint with
                    # tolerance
                    prob += plp.lpSum(dev_terms) <= tolerance, "Dev_{}_{}_{}".format(
                        cur_bid, dev_bid, bidder_id)

    # Constraint that Pr[v | b] is a distribution for each fixed b
    for bids in bid_pdf.keys():
        cond_vars = [psi_vars[(value, bids)] for value in range(max_value + 1)]
        prob += plp.lpSum(cond_vars) == 1, "Density_Psi_{}".format(bids)

    # Objective coefficients based on moment function for the lower bound
    # Minimize sum_{v, b} f(v) * Pr[v | b] * phi(b)
    prob += plp.lpSum([moment_fn(v) * bid_pdf[bids] * psi_vars[(v, bids)]
                       for (v, bids) in product(*[range(max_value + 1), bid_pdf.keys()])])

    # Solve the LP
    prob.solve(get_solver(solver_str))
    # Get the objective value of the LP which is a lower bound on the moment
    if int(prob.status) == 1:
        lower_bound = plp.value(prob.objective)
    else:
        lower_bound = None

    # Now we compute the upper bound on the moment, i.e.
    # Minimize - sum_{v,b} f(v) * Pr[v | b] * phi(b)
    prob += plp.lpSum([- moment_fn(v) * bid_pdf[bids] * psi_vars[(v, bids)]
                       for (v, bids) in product(*[range(max_value + 1), bid_pdf.keys()])])

    # Solve the LP
    prob.solve(get_solver(solver_str))
    # Get minus the objective value of the LP which is an upper bound on the
    # moment
    if int(prob.status) == 1:
        upper_bound = -plp.value(prob.objective)
    else:
        upper_bound = None

    return (lower_bound, upper_bound)


def inverse_bce_parameterized(num_bidders, max_value, bid_pdf, moment_fn, density, tolerance=0.0001, solver_str='COIN'):
    """Computes a sharp identified set of a moment of the common value
    distribution from an observed bid vector distribution. It constructs
    the following inverse BCE-LP:

    max_{Pr[v | b]} sum_{v in {0,...,max_value}, b in S} f(v) * Pr[v | b] * phi(b) \n
    forall i, b_i*, b_i': \n
    sum_{b in S: b_i=b_i*, v in {0,...,max_value}} \n
                Pr[v | b] * phi(b) * (U_i(b_i',b_{-i};v)-U_i(b;v)) <= tolerance \n
    forall v: sum_{v in {0,...,max_value}} Pr[v | b] = 1 \n

    Parameters
    ----------
    - num_bidders : (int) the number of bidders
    - max_value : (int) the maximum value/bid. Values and bids are constraint in {0,1,...,max_value}
    - bid_pdf : (dictionary from tuple of ints to float) the pdf phi of the bid vector
        distribution as a dictionary with keys the tuple of bids and value the pdf of
        that tuple of bids
    - moment_fn : (function from float to float) the moment function
    - tolerance : (float) a tolerance on the best response constraints in case of sampling
        or numerical error in the bid_pdf

    Returns
    -------
    - (lower_bound, upper_bound) : (double, double) the upper and lower bound of the
      sharp identified set of the moment
    """

    # create an LP instance
    prob = plp.LpProblem("Inverse-BCE", plp.LpMinimize)

    # Create the psi variables which correspond to Pr[v | b] for each (v,b)
    # pair, where b is in the support of the bid_pdf
    lp_var_keys = product(*[range(max_value + 1), bid_pdf.keys()])
    psi_vars = plp.LpVariable.dicts('psi', lp_var_keys, lowBound=0)

    # Creating the best response constraints with tolerance
    devs = product(
        *[range(max_value + 1), range(max_value + 1), range(0, num_bidders)])
    for (cur_bid, dev_bid, bidder_id) in devs:  # for all b_i*, b_i', i
        if cur_bid != dev_bid:  # if b_i* \neq b_i'
            # Create all the terms of the form Pr[v | b] * phi(b) * (U_i' - U_i)
            # for all b \in S, such that b_i = b_i*
            dev_terms = [
                deviation_term(dev_bid, bidder_id, v, bids,
                               psi_vars[(v, bids)], bid_pdf[bids])
                for (v, bids) in product(*[range(max_value + 1), bid_pdf.keys()])
                if bids[bidder_id] == cur_bid]
            # Add these terms to create the best response constraint with
            # tolerance
            prob += plp.lpSum(dev_terms) <= tolerance, "Dev_{}_{}_{}".format(
                cur_bid, dev_bid, bidder_id)

    # Constraint that Pr[v | b] is a distribution for each fixed b
    for bids in bid_pdf.keys():
        cond_vars = [psi_vars[(value, bids)] for value in range(max_value + 1)]
        prob += plp.lpSum(cond_vars) == 1, "Density_Psi_{}".format(bids)

     # Constraint that $\Pi(v)$ is consistent with "density"
    for value in range(max_value + 1):
        density_vars = [bid_pdf[bids] *
                        psi_vars[(value, bids)] for bids in bid_pdf.keys()]
        prob += plp.lpSum(density_vars) <= density[
            value] + tolerance*(1/max_value), "Density_Pi_Upper_{}".format(value)
        prob += plp.lpSum(density_vars) >= density[
            value] - tolerance*(1/max_value), "Density_Pi_Lower_{}".format(value)

    # Objective coefficients based on moment function for the lower bound
    # Minimize sum_{v, b} f(v) * Pr[v | b] * phi(b)
    prob += 0

    # Solve the LP
    prob.solve(get_solver(solver_str))
    # Get the objective value of the LP which is a lower bound on the moment
    if int(prob.status) == 1:
        lower_bound = plp.value(prob.objective)
    else:
        lower_bound = None

    return lower_bound


def inverse_bce_parameterized_sparse(num_bidders, max_value, bid_pdf, moment_fn, density, tolerance=0.0001, solver_str='COIN'):
    """Computes a sharp identified set of a moment of the common value
    distribution from an observed bid vector distribution. This method takes advantage
    of sparsity in the support of the BCE, by only creating deviating bids that
    are undominated, given the BCE distribution.

    It constructs the following inverse BCE-LP:

    max_{Pr[v | b]} sum_{v in {0,...,max_value}, b in S} f(v) * Pr[v | b] * phi(b) \n
    forall i, b_i*, b_i' that are undominated: \n
    sum_{b in S: b_i=b_i*, v in {0,...,max_value}} \n
                Pr[v | b] * phi(b) * (U_i(b_i',b_{-i};v)-U_i(b;v)) <= tolerance \n
    forall v: sum_{v in {0,...,max_value}} Pr[v | b] = 1 \n

    Parameters
    ----------
    - num_bidders : (int) the number of bidders
    - max_value : (int) the maximum value/bid. Values and bids are constraint in {0,1,...,max_value}
    - bid_pdf : (dictionary from tuple of ints to float) the pdf phi of the bid vector
        distribution as a dictionary with keys the tuple of bids and value the pdf of
        that tuple of bids
    - moment_fn : (function from float to float) the moment function
    - tolerance : (float) a tolerance on the best response constraints in case of sampling
        or numerical error in the bid_pdf

    Returns
    -------
    - (lower_bound, upper_bound) : (double, double) the upper and lower bound of the
      sharp identified set of the moment
    """

    # create an LP instance
    prob = plp.LpProblem("Inverse-BCE", plp.LpMinimize)

    # Create the psi variables which correspond to Pr[v | b] for each (v,b)
    # pair, where b is in the support of the bid_pdf
    lp_var_keys = product(*[range(max_value + 1), bid_pdf.keys()])
    psi_vars = plp.LpVariable.dicts('psi', lp_var_keys, lowBound=0)

    # Creating the best response constraints with tolerance
    for bidder_id in range(num_bidders):
        # Find what are all the bids the player ever submits under BCE
        cur_bid_support = set([b_vec[bidder_id] for b_vec in bid_pdf.keys()])
        for cur_bid in cur_bid_support:
            # Find the possible values that the maximum other bid can take
            max_other_bid_support = set(
                [max(b_vec[0:bidder_id] + b_vec[bidder_id + 1:]) for b_vec in bid_pdf.keys()
                 if b_vec[bidder_id] == cur_bid])
            dev_bid_support = max_other_bid_support | set(
                [b + 1 for b in max_other_bid_support]) | set([0])
            for dev_bid in dev_bid_support:
                if cur_bid != dev_bid:  # if b_i* \neq b_i'
                    # Create all the terms of the form Pr[v | b] * phi(b) * (U_i' - U_i)
                    # for all b \in S, such that b_i = b_i*
                    dev_terms = [
                        deviation_term(dev_bid, bidder_id, v, bids,
                                       psi_vars[(v, bids)], bid_pdf[bids])
                        for (v, bids) in product(*[range(max_value + 1), bid_pdf.keys()])
                        if bids[bidder_id] == cur_bid]
                    # Add these terms to create the best response constraint with
                    # tolerance
                    prob += plp.lpSum(dev_terms) <= tolerance, "Dev_{}_{}_{}".format(
                        cur_bid, dev_bid, bidder_id)

    # Constraint that Pr[v | b] is a distribution for each fixed b
    for bids in bid_pdf.keys():
        cond_vars = [psi_vars[(value, bids)] for value in range(max_value + 1)]
        prob += plp.lpSum(cond_vars) == 1, "Density_Psi_{}".format(bids)

     # Constraint that $\Pi(v)$ is consistent with "density"
    for value in range(max_value + 1):
        density_vars = [bid_pdf[bids] *
                        psi_vars[(value, bids)] for bids in bid_pdf.keys()]
        prob += plp.lpSum(density_vars) <= density[
            value] + tolerance*(1/max_value), "Density_Pi_Upper_{}".format(value)
        prob += plp.lpSum(density_vars) >= density[
            value] - tolerance*(1/max_value), "Density_Pi_Lower_{}".format(value)

    # Objective coefficients based on moment function for the lower bound
    # Minimize sum_{v, b} f(v) * Pr[v | b] * phi(b)
    prob += plp.lpSum([moment_fn(v) * bid_pdf[bids] * psi_vars[(v, bids)]
                       for (v, bids) in product(*[range(max_value + 1), bid_pdf.keys()])])

    # Solve the LP
    prob.solve(get_solver(solver_str))

    # Get the objective value of the LP which is a lower bound on the moment
    if int(prob.status) == 1:
        lower_bound = plp.value(prob.objective)
    else:
        lower_bound = None

    return lower_bound


def inverse_bce_parameterized_sparse_min_tolerance(num_bidders, max_value, bid_pdf, moment_fn, density, solver_str='COIN'):
    """Computes a sharp identified set of a moment of the common value
    distribution from an observed bid vector distribution. This method takes advantage
    of sparsity in the support of the BCE, by only creating deviating bids that
    are undominated, given the BCE distribution.

    It constructs the following inverse BCE-LP:

    max_{Pr[v | b]} sum_{v in {0,...,max_value}, b in S} f(v) * Pr[v | b] * phi(b) \n
    forall i, b_i*, b_i' that are undominated: \n
    sum_{b in S: b_i=b_i*, v in {0,...,max_value}} \n
                Pr[v | b] * phi(b) * (U_i(b_i',b_{-i};v)-U_i(b;v)) <= tolerance \n
    forall v: sum_{v in {0,...,max_value}} Pr[v | b] = 1 \n

    Parameters
    ----------
    - num_bidders : (int) the number of bidders
    - max_value : (int) the maximum value/bid. Values and bids are constraint in {0,1,...,max_value}
    - bid_pdf : (dictionary from tuple of ints to float) the pdf phi of the bid vector
        distribution as a dictionary with keys the tuple of bids and value the pdf of
        that tuple of bids
    - moment_fn : (function from float to float) the moment function
    - tolerance : (float) a tolerance on the best response constraints in case of sampling
        or numerical error in the bid_pdf

    Returns
    -------
    - (lower_bound, upper_bound) : (double, double) the upper and lower bound of the
      sharp identified set of the moment
    """

    # create an LP instance
    prob = plp.LpProblem("Inverse-BCE", plp.LpMinimize)

    # Create the psi variables which correspond to Pr[v | b] for each (v,b)
    # pair, where b is in the support of the bid_pdf
    lp_var_keys = product(*[range(max_value + 1), bid_pdf.keys()])
    psi_vars = plp.LpVariable.dicts('psi', lp_var_keys, lowBound=0)

    tolerance_var = plp.LpVariable('tolerance', lowBound=0)

    # Creating the best response constraints with tolerance
    for bidder_id in range(num_bidders):
        # Find what are all the bids the player ever submits under BCE
        cur_bid_support = set([b_vec[bidder_id] for b_vec in bid_pdf.keys()])
        for cur_bid in cur_bid_support:
            # Find the possible values that the maximum other bid can take
            max_other_bid_support = set(
                [max(b_vec[0:bidder_id] + b_vec[bidder_id + 1:]) for b_vec in bid_pdf.keys()
                 if b_vec[bidder_id] == cur_bid])
            dev_bid_support = max_other_bid_support | set(
                [b + 1 for b in max_other_bid_support]) | set([0])
            for dev_bid in dev_bid_support:
                if cur_bid != dev_bid:  # if b_i* \neq b_i'
                    # Create all the terms of the form Pr[v | b] * phi(b) * (U_i' - U_i)
                    # for all b \in S, such that b_i = b_i*
                    dev_terms = [
                        deviation_term(dev_bid, bidder_id, v, bids,
                                       psi_vars[(v, bids)], bid_pdf[bids])
                        for (v, bids) in product(*[range(max_value + 1), bid_pdf.keys()])
                        if bids[bidder_id] == cur_bid]
                    # Add these terms to create the best response constraint with
                    # tolerance
                    prob += plp.lpSum(dev_terms) <= tolerance_var, "Dev_{}_{}_{}".format(
                        cur_bid, dev_bid, bidder_id)

    # Constraint that Pr[v | b] is a distribution for each fixed b
    for bids in bid_pdf.keys():
        cond_vars = [psi_vars[(value, bids)] for value in range(max_value + 1)]
        prob += plp.lpSum(cond_vars) == 1, "Density_Psi_{}".format(bids)

    # Constraint that $\Pi(v)$ is consistent with "density"
    for value in range(max_value + 1):
        density_vars = [bid_pdf[bids] *
                        psi_vars[(value, bids)] for bids in bid_pdf.keys()]
        prob += plp.lpSum(density_vars) <= density[
            value] + tolerance_var*(1/max_value), "Density_Pi_Upper_{}".format(value)
        prob += plp.lpSum(density_vars) >= density[
            value] - tolerance_var*(1/max_value), "Density_Pi_Lower_{}".format(value)


    # Objective coefficients based on moment function for the lower bound
    # Minimize sum_{v, b} f(v) * Pr[v | b] * phi(b)
    prob += tolerance_var

    # Solve the LP
    prob.solve(get_solver(solver_str))

    # Get the objective value of the LP which is a lower bound on the moment
    if int(prob.status) == 1:
        min_tolerance = plp.value(prob.objective)
    else:
        min_tolerance = -1

    return min_tolerance


def unit_tests():
    """ Unit tests for the functions in this module """
    import utils
    import parse_file

    nb_bidders = 2
    max_value = 20

    bid_pdf = {}
    bid_pdf[(0, 0)] = 0.5
    bid_pdf[(100, 100)] = 0.5

    # When the tolerance is 0, should be infeasible
    tol = 0
    lower_bound, upper_bound = inverse_bce_sparse(
        nb_bidders, max_value, bid_pdf, utils.first_moment, tolerance=tol)
    if not lower_bound and not upper_bound:
        print("Infeasibility test: passed")
    else:
        print("Infeasibility test: failed")

    # When the tolerance is infinite, we should get the whole range
    tol = 10**6
    lower_bound, upper_bound = inverse_bce_sparse(
        nb_bidders, max_value, bid_pdf, utils.first_moment, tolerance=tol)
    if lower_bound == 0 and upper_bound == max_value:
        print("Tolerance test: passed")
    else:
        print("Tolerance test: failed")

    # The sparse and not sparse version should give the same solutions on reasonable examples.
    # Using the data with max_bid=5, max_value=10 as an example here.
    max_value = 20
    max_bid = 10

    wb = openpyxl.load_workbook('Dataset.xlsx')
    sheet1 = wb.get_sheet_by_name('Tract79')
    sheet2 = wb.get_sheet_by_name('Trbid79')

    bid_pdf = parse_file.bins_from_data(wb, sheet1, sheet2, nb_bidders, max_bid)

    tol = 0.1
    lower_bound_sparse, upper_bound_sparse = inverse_bce_sparse(
        nb_bidders, max_value, bid_pdf, utils.first_moment, tolerance=tol)
    lower_bound, upper_bound = inverse_bce(
        nb_bidders, max_value, bid_pdf, utils.first_moment, tolerance=tol)

    if abs(upper_bound_sparse - upper_bound) < 0.00001 and abs(lower_bound_sparse - lower_bound) < 0.00001:
        print("Equality test: passed")
    else:
        print("Equality test: failed")

    # The parametrized sparse code for min tolerance should return 0 when the data is generated using compute_bce
    mu = 5
    sigma = 5
    density = [np.exp(-(v - mu)**2 / (2 * sigma**2))
                       for v in range(max_value + 1)]
    bid_pdf = compute_bce(nb_bidders, max_value, density)
    min_tol = inverse_bce_parameterized_sparse_min_tolerance(nb_bidders, 
                       max_value, bid_pdf, utils.first_moment, density)
    if min_tol <= 10**(-5):
        print("Minimum tolerance test: passed")
    else:
        print("Minimum tolerance test: failed")
        print("Minimum tolerance: " + str(min_tol))

    bid_pdf = {}
    bid_pdf[(0, 0)] = 0.5
    bid_pdf[(100, 100)] = 0.5

    # When the tolerance is 0, the parameterized codes should be infeasible
    tol = 0
    lower_bound = inverse_bce_parameterized_sparse(
        nb_bidders, max_value, bid_pdf, utils.first_moment, density, tolerance=tol)
    lower_bound_sparse = inverse_bce_parameterized_sparse(
        nb_bidders, max_value, bid_pdf, utils.first_moment, density, tolerance=tol)
    if not lower_bound and not lower_bound_sparse:
        print("Parameterized infeasibility test: passed")
    else:
        print("Parameterized infeasibility test: failed")

    # When the tolerance is infinite, we should get the whole range for the parameterized code
    tol = 10**6
    lower_bound = inverse_bce_parameterized_sparse(
        nb_bidders, max_value, bid_pdf, utils.first_moment, density, tolerance=tol)
    lower_bound_sparse = inverse_bce_parameterized_sparse(
        nb_bidders, max_value, bid_pdf, utils.first_moment, density, tolerance=tol)
    if lower_bound == 0 and lower_bound_sparse == 0:
        print("Parameterized tolerance test: passed")
    else:
        print("Parameterized tolerance test: failed")

if __name__ == '__main__':
    unit_tests()
