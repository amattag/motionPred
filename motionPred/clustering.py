"""
Clustering with the Expectation-Maximization (E-M) algorithm

Demonstration of how to cluster a set of trajectories using the 
Expectation-Maximization (E-M) algorithm.

Usage: >python clustering.py $tjcsN $clustersN $maxIter $tol
"""

# Author: Antonio Matta <antonio.matta@upm.es>

# Import required modules.
import numpy as np
import matplotlib.pyplot as plt
from os import path
import time
import sys
import random

# import auxiliar modules.
import expMax as em
import plotting as myplt
import probability as pobty

# Check if the correct input is given in command-line arguments.
if len(sys.argv) != 5:
	print('Usage: ./clustering.py $tjcsN $clustersN $maxIter $tol')
	sys.exit(1)         # Return a non-zero value for abnormal termination

# Load arguments into variables.
print 'Argument List:', str(sys.argv)
tjcsN = int(sys.argv[1])     # Number of Trajectories to load.
clustersN = int(sys.argv[2]) # Number of Clusters.
maxIter = int(sys.argv[3])   # Maximum number of iterations for the E-M algoritm.
tol = float(sys.argv[4])     # error tolerance for the E-M algorithm.

# Output input arguments.
print "Number of Trajectories:", tjcsN
print "Number of Clusters:", clustersN
print "E-M iterations:", maxIter
print "E-M Converge Threshold:", tol

# Load trayectory data from the 'tjcs.npy' file.
# A set of 1856 trajectories generated from the Edinburgh Informatics Forum 
# Pedestrian Database: http://homepages.inf.ed.ac.uk/rbf/FORUMTRACKING/
data = np.load("data/tjcs.npy")

# Choose the first number of trajectories given by the tjcsN argument.
tjcs = data[:tjcsN]

# Plot selected trajectories
fig1 = plt.figure()
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("600x500+200+0")
fig1 = plt.title("Trajectories")
plt.axis([0,16,0,12])
plt.xlabel('X (mts)')
plt.ylabel('Y (mts)')
myplt.plot_trajectories(tjcs)
plt.pause(0.05)

# Get the length of each trajectory.
tjcsLength = tjcs.shape[1]
   
# Select intial means for E-M algorithm.
means = np.zeros((clustersN, tjcsLength, 2))
meansIndex = random.sample(np.arange(tjcsN), clustersN)
for m, n in zip( xrange(clustersN), meansIndex):
	means[m] = data[n]

# Select intial means and best number of clusters.
#means, meansIndexTmp = em.selectClusters(tjcs, tjcsN, tjcsLength)
#clustersN = means.shape[0]
#print "clustersN", clustersN
#meansIndex = [i[0] for i in meansIndexTmp]

# Compute covariance Matrix.
vtjcs=tjcs.reshape((tjcsN*tjcsLength, tjcs.shape[2]))
covariance = np.round(np.cov(vtjcs.T)*np.eye(vtjcs.shape[1]))

### Let's iterate with the E-M algorithm ###
ll_old = 0
i = 0
while i < maxIter:
	# E-M algorithm.
    clusters = em.expectation(tjcs, means, covariance, pobty.t_gaussian)
    em.maximization(tjcs, clusters, means, pobty.t_zero)
    
    # Update log likelihoood.
    ll_new = 0.0
    for j in range(tjcsN):
		s = 0
		for k in range(clustersN):
			s += pobty.t_gaussian(means[k], covariance, tjcs[j])
		ll_new += np.log(s)
		
	# If the threshold is below the expected tol factor.
    if np.abs(ll_new - ll_old) < tol:
		# If found, replace the worst cluster with the worst represented trajectory 
		# in order to improve the quality of the returned clusters.
		
		# Finds worst cluster and its index.
		clusterIndex, clusterScore = em.worst_cluster(clusters)
		
		# Finds the index of the worst represented trajectory.
		tjcIndex = em.worst_trajectory(clusters, clusterIndex, clusterScore, meansIndex, tjcs, covariance)
		
		# If the worst represented trajectory score is higher than the worst cluster score,
		# replace cluster with the found trajectory.
		if tjcIndex == -1:
			break
		else:
			print "Replacing cluster %i with trajectory %i" % (clusterIndex, tjcIndex)
			means[clusterIndex] = tjcs[tjcIndex]
			meansIndex.append(tjcIndex)
			ll_old = 0
			i = 0
			continue
    else:
		ll_old = ll_new
		i += 1
		 
    print "Iteration Number: ", i 
    
# This is to plot the results of the E-M algorithm.
fig2 = plt.figure()
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("600x500+850+0")
plt.axis([0,16,0,12])
plt.xlabel('X (mts)')
plt.ylabel('Y (mts)')
plt.title("Clustering, Iteration Number: %s" %(i+1))
myplt.plot_clusters(clusters, tjcs)

# Save means structre, which represents the clusters typical trayectories into
# a npy file.
timestr = time.strftime("%Y%m%d-%H%M%S")
filename=path.join("data/"+timestr+"-clusters")
np.save(filename, means)
    
# Let's plot the found clusters.
fig3 = plt.figure()
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("600x500+200+600")
fig3 = plt.title("Clusters")
plt.grid()
plt.xticks(np.arange(0, 16, 1.0))
plt.yticks(np.arange(0, 12, 1.0))
plt.xlabel('X (mts)')
plt.ylabel('Y (mts)')
fig3 = myplt.plot_time_model(means)
#fig3 = myplt.plot_trajectories(means)

# Plot cluster contribs
cluster_contribs = np.sum(clusters, 0)
fig4 = plt.figure()
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("600x500+850+600")
fig4 = plt.title("Number of Trajectories per Cluster")
fig4 = plt.bar(np.arange(clustersN), cluster_contribs, color = myplt.generate_palette(clustersN))
plt.show()
