import os
import random
import openpyxl
import numpy as np
import math



def first_moment(value):
    """ The expected value, which is the identity. """
    return value


def second_moment(value):
    """ Returns the second moment. """
    return value**2


def brooks_bound(nb_bidders, max_value, bid_pdf):
    """ Computes the upper bound on the mean the is implied by Brooks et al.
    paper.
    """
    revenue = sum([max(bids) * bid_pdf[bids] for bids in bid_pdf])
    if nb_bidders == 2:
        return 2 * math.sqrt(revenue * max_value) - revenue
    else:
        return math.sqrt(2 * nb_bidders * revenue * max_value / (nb_bidders - 1))



if __name__ == '__main__':
    """ Add some unit test for these utilt files """
