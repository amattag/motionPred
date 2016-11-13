"""
Expectation-Maximization module.

Implementation of the Expectation-Maximization algorithm to cluster data.
"""

# Author: Antonio Matta <antonio.matta@upm.es>

# Import required modules.
import numpy as np
from scipy import linalg
import random

# import auxiliar modules.
import probability as pobty

# BIC Criterion
def computeBIC(tjcs, means, tjcsN, clustersN):
	""" Compute the The Bayesian Information criterion (BIC) of a given 
	    model.
	    
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
	x = np.zeros(clustersN)
	
	# Compute the error variance for the set of observations.
	for n in xrange(clustersN):
		centroid = np.sum(means[n], 0) / means[n].shape[0]
		for m in xrange(tjcsN):
			# Error variance function.
		    diff = tjcs[m] - centroid
		    errVar[n, m] = np.square(linalg.norm(diff)) 
		x[n] = np.sum(errVar[n, :]) / clustersN
		
	# Total error variance of the model.	    
	err = np.sum(x)
	
	# Log likelihood of the model. 
	R = clustersN*tjcsN*means[n].shape[0]
	maxLike = err / (2*(R-clustersN))
	
	# Compute the BIC Criterion value6.
	bic = tjcsN*np.log(maxLike) + clustersN*np.log(tjcsN)
	
	return bic
	
# Select the clusters and number of clusters based on the BIC criterion.
def selectClusters(tjcs, tjcsN, tjcsLength):
    """This function selects the number of proper clusters for a given
       model based on the BIC criterion function.
    
    Parameters
    ----------
    tjcs: array
      Array with the set of trajectories.
      
    tjcsN: int
      Number of trajectories.
      
    tjcsLenght: int
      Lenght of each trajectory.
      
    Returns
    -------
    clusters: array
      An array with the proper number of clusters.
    """
    bicOld = 0
    meansIndex = [] # List to store the indexes of the found clusters.
    for i in xrange(tjcsN):
		# Fill means structure by selecting a trajectory at random.
		tjcIndex = 0
		clustersN = 1
		tjcIndex = random.sample(np.arange(tjcsN), clustersN)
		meansIndex.append(tjcIndex)	
		if (i == 0):
			means = np.zeros((clustersN, tjcsLength, 2)) # Means matrix.
			means = tjcs[tjcIndex]		
		else:	
			meansTmp = np.zeros((clustersN, tjcsLength, 2)) # Means matrix.
			meansTmp = tjcs[tjcIndex]
			means = np.vstack((means, meansTmp))
			
		# Compute BIC criterion value.
		bicNumber = 0
		clustersN = means.shape[0] 
		bicNumber = computeBIC(tjcs, means, tjcsN, clustersN)
		
		# If new BIC value is bigger, return means structure and break.
		if ((clustersN != 1) and (bicNumber > bicOld)) :
			return means, meansIndex
			break
			
		bicOld = bicNumber
		
		
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


def maximization(tjcs, clusters, means, zero):
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
    """ Find the worst cluster: the cluster tha contributes the least to
    the solution. This is to exclude similar motion patterns.
    
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
    clustersN = clusters.shape[1]  # Number of Clusters.
    currentClusters = np.sum(np.max(clusters, 1)) # Sum of best cluster scores.
    clusterScores = np.zeros((clustersN, ))
    
    # Find the score of every cluster.
    for m in xrange(clustersN):
        withoutCluster = clusters.copy()
        withoutCluster[:, m] = 0. # Exclude current cluster.
        clusterScores[m] = currentClusters - np.sum(np.max( withoutCluster, 1))
    
    # The cluster that scores the less is the selected one.    
    clusterIndex = np.argmin(clusterScores)
    clusterScore = np.min(clusterScores)
    return clusterIndex, clusterScore


def worst_trajectory(clusters, clusterIndex, clusterScore, selected, tjcs, covariance):
    """ Find the worst represented trajectory: For the worst trajectory, 
    we iterate with individual trajectory scores, replacing them with the worst cluster 
    and looking for a positive increase in the cluster score.
    
    If the score is bigger, return the trajectory index, because it is the
    worst represented one. 
    
    Parameters
    ----------
    clusters: array like
      An array containing all the clusters.
      
    clusterIndex: int
      index of the worst cluster.
      
    clusterScore: float
      score of the worst cluster.
      
    selected: array
      An array with the indices of the already selected clusters.
    
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
          
    If fail:
      -1: int
        Error message.
    """
    tjcsN = tjcs.shape[0]      # Number of Trajectories.
    
    # Sort trajectories by contribution: from least to greatest.
    tjcsContribs = np.sum(clusters, 1)
    tjcsContribs[selected] = 1E6    # Ignore trajectories already selected.
    sortedTjcs = np.argsort(tjcsContribs)
    
    # Structures to store cluster scores.
    newClusters = clusters.copy()  
    newClusters[:, clusterIndex] = 0.0    # Exclude current cluster.
    currentClusters = np.sum(np.max(clusters, 1))  # Sum of best cluster scores. 
    
    # Find new cluster score with worst trajectory. 
    for k in sortedTjcs[:(len(sortedTjcs)/10)]:
		for n in xrange(tjcsN):
			# Call the Expectation step using worst trajectory as mean.
			newClusters[n, clusterIndex] = pobty.t_gaussian(tjcs[k], covariance, tjcs[n])
		newClusters[n, clusterIndex] /= np.sum(newClusters[:, clusterIndex])  # Normalize
		
		# Difference between cluster with worst Tjc and the old cluster.
		tjcScore = np.sum(np.max(newClusters, 1)) - currentClusters
		
		# If the new cluster score is bigger than the worst cluster one, 
		# return the index of the tractory.
		if tjcScore > clusterScore:
			return k 
			
	# If a worse trajectory score is not found, return error message.
    return -1
