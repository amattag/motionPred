"""
Expectation-Maximization module.

Implementation of the Expectation-Maximization algorithm to cluster data.
"""

# Author: Antonio Matta <antonio.matta@upm.es>

# Import required modules.
import numpy as np

# import auxiliar modules.
import probability as pobty

# BIC Criterion
def computeBIC(tjcs, means, tjcsN, clustersN):
	""" Compute the BIC criterion of a given model.
	
	Parameters
	----------
	tjcs: array
      Array with the set of trajectories of a given model.
      
    means: array
      Array with  the trajectories used as means.
      
    tcsN: int
      Number of trajectories.
      
    clusterN: int
      Number of clusters.    
	
	Returns
	-------
	bic: float
	  A number storing the BIC criterion value. This value help us select
	  the best number of clusters for a given model.
	"""
	
	errVar = np.zeros((clustersN, tjcsN))  # A matrix that stores the error variance.
	
	# Compute the error variance for the set of observations.
	for n in xrange(clustersN):
		for m in xrange(tjcsN):
			# Erro variance function.
		    diff = tjcs[m] - means[n]
		    errVar[n, m] = np.sum(np.square(diff)) / tjcsN
		errVar[n, m] = np.sum(errVar[n, :]) / clustersN 
	
	# Total error variance of the model.	    
	err = np.sum(errVar) / clustersN
	
	# Log likelihood of the model.
	maxLike = err / tjcsN
	
	# Compute the BIC Criterion value.
	bic = tjcsN*np.log(maxLike) + clustersN*np.log(tjcsN)
	
	return bic
	
	
# Expectation-Maximization algorithm.
def expectation(tjcs, means, covariance, gaussian):
    """Expectation step: For every trajectory calculate the Expected likehood
    that it belongs to the current cluster model given by the means array.
    
    Parameters
    ----------
    tjcs: array
      Array with a set of trajectories to train the E-M algorithm.
      
    means: array
      Array with  the trajectories used as means.
      
    covariance: matrix array
      Diagonal matrix with the covariance data.
      
    gaussian: function
      A function that computes the Multivariate Gaussian Probability Distribution
      Function (PDF). 
  
    Returns
    -------
    clusters: array
      An array with the clusters found on every step of the E-M algorithm.
    """
    tjcsN = len(tjcs)                   # Number of trajectories
    cltsN = len(means)                  # Number of clusters
    clusters = np.zeros((tjcsN, cltsN)) # Expected clusters
    for n in xrange(tjcsN):
        for m in xrange(cltsN):
            # Call to Multivariate Gaussian PDF.
            clusters[n, m] = gaussian(means[m], covariance, tjcs[n])
        clusters[n, m] /= np.sum(clusters[n, :])  # Normalize
    return clusters


def maximization(tjcs, clusters, means, zero, cummulate):
    """Maximization step: Calculate new cluster models that Maximizes the 
    Expected likehood.
    
    Parameters
    ----------
    tjcs: array
      An array with a set of trajectories to train the E-M algorithm.
    
    clusters: array
      An array with the clusters found on every step of the E-M algorithm.
      
    means: array
      Array with  the trajectories used as means.
    
    zero: function
      A function that sets an array to zero.
    
    cummulate: function
      A function to weight data.
    
    Returns
    -------
    It Maximizes the expected likehood of the clusters array.
    """
    tjcsN = len(tjcs)
    cltsN = len(means)
    
    # Updating the means values.
    for m in xrange(cltsN):
        zero(means[m])
        for n in xrange(tjcsN):
			means[m] += clusters[n, m]*tjcs[n]
        means[m] /= np.sum(clusters[:, m]) # Normalize
        
            
# Functions to optimize the quality of the clusters.
def worst_cluster(clusters):
    """ Find the worst cluster: the cluster with the fewer number of
        trajectories.
    
    Parameters
    ----------
    clusters: array
      An array containing all the clusters.
    
    Returns
    -------
    index: int
      index of the worst cluster.
    
    score: float
      score of the worst cluster.
    """
    M = clusters.shape[1]  # Number of Clusters.
    with_cluster = np.sum(np.max(clusters, 1)) # Sum of best cluster scores.
    cluster_scores = np.zeros((M, ))
    for m in xrange(M):
        without_cluster = clusters.copy()
        without_cluster[:, m] = 0. # Exclude current cluster.
        cluster_scores[m] = with_cluster - np.sum(np.max( without_cluster, 1))
    # The cluster that scores the less is the selected one.    
    index = np.argmin(cluster_scores)
    score = np.min(cluster_scores)
    return index, score


def worst_trajectory(clusters, c_index, c_score, visited, tjcs, covariance):
    """ Find the worst trajectory: For the worst trajectory, we iterate with 
    individual trajectory scores, replacing them with the worst cluster and 
    looking for a positive increase in the score.
    
    If the score is bigger, return the trajectory index and its score. 
    
    Parameters
    ----------
    clusters: array like
      An array containing all the clusters.
      
    c_index: int
      index of the worst cluster.
      
    c_score: float
      score of the worst cluster.
      
    visited: array
      An array with the indices of the current clusters.
    
    tjcs: array
      An array with the trayectories used to train the E-M algorithm and to 
      obtain the clusters.
    
    covariance: matrix array
      A matrix with the covariance data.
  
    Returns
    -------
    If success:
      k: int
        index of the worst trajectory.
        
      t_score: float
        score of the worst trajectory.
        
    If fail:
      -1: int
        Error message.
        
       0: int
         Error message.  
    """
    N = tjcs.shape[0]      # Number of Trajectories.
    M = clusters.shape[1]  # Number of Clusters.
    
    # Sort trajectories by contribution.
    traj_contribs = np.sum(clusters, 1)
    traj_contribs[visited] = 1E6 # Ignore trajectories already visited.
    sorted_trajs = np.argsort(traj_contribs)
    
    # Find worst trajectory. If its score is bigger than the worst cluster one,
    # return its score and index.
    tmp_cluster = clusters.copy()
    tmp_cluster[:, c_index] = 0.0
    without_cluster = np.sum(np.max(clusters, 1)) 
    for k in sorted_trajs:
        for n in xrange( N ):
            # Call to Multivariate Gaussian PDF.
            tmp_cluster[n, c_index] = pobty.t_gaussian(tjcs[k], covariance, tjcs[n])
        t_score = np.sum(np.max(tmp_cluster, 1)) - without_cluster
        if t_score > c_score:
            return k, t_score
    return -1, 0.0
