"""
Probability module

A module with typical probability functions used in robotics for clustering
techniques.
"""

# Author: Antonio Matta <antonio.matta@upm.es>

# Import required modules.
import numpy as np
from scipy import linalg

# Auxiliar functions: t_gaussian, t_zero and t_cummulate functions.
def t_zero( trajectory ):
    """ Set the values of a trajectory to zero.
    
    Parameters:
    ----------
    trajectory: array
      A set of 2D (x, y) points. Each point of is a float number.
    """
    trajectory *= 0.0


def t_cummulate( t1, weight, t2 ):
    """ Cummulate weighted tjc t2 into t1
    
    Parameters:
    -----------
    weight: array
      Array with weight values.
      
    t1: array
      Array to be weighted.
    
    t2: array
      Array use to weight another array.
    """
    t1 += weight * t2
  
        
def t_gaussian( mean, covariance, value ):
    """ Multivariate Gaussian Probability Distribution Function (PDF)
    
    Parameters:
      mean: array
        Array with  the trajectories used as means.
         
      covariance: matrix array
        Diagonal matrix with the covariance data.
        
      value: array
        Trajectories data.
    
    Return:
      exp: array
        An array of expected clusters. 
    """
    inv = linalg.inv(covariance)
    diff = mean - value
    dist = -0.5 * np.dot(np.dot(diff, inv), diff.transpose())
    exp = np.exp(np.diagonal(dist))
    return np.multiply.reduce(exp)
