"""
Clustering with the Expectation-Maximization (E-M) algorithm

Demonstration of how to cluster a set of trajectories using the 
Expectation-Maximization (E-M) algorithm.
"""

# Author: Antonio Matta <antonio.matta@upm.es>

# Import required modules.
import numpy as np
import matplotlib.pyplot as plt

# import auxiliar modules.
import expMax as em
import plotting as myplt
import probability as pobty

# Load trayectory data from the 'tjcs.npy' file.
# A set of 1856 trajectories generated from the Edinburgh Informatics Forum 
# Pedestrian Database: http://homepages.inf.ed.ac.uk/rbf/FORUMTRACKING/
data = np.load("data/tjcs.npy")

# Choose the first 500 trajectories.
tjcs = data[:200]

# Plot selected trajectories
fig1 = plt.figure()
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("600x500+50+50")
fig1 = plt.title("Trajectories")
plt.axis([0,16,0,12])
plt.xlabel('X (mts)')
plt.ylabel('Y (mts)')
myplt.plot_trajectories(tjcs)
plt.pause(0.05)

# Get number of trajectories and the lenght of every of them.
tjcsN = tjcs.shape[0]         # Number of Trajectories. 
tjcsLength = tjcs.shape[1]    # Lenght of each trajectory.
   
# Select intial means and best number of clusters.
means = em.selectClusters(tjcs, tjcsN, tjcsLength)
clustersN = means.shape[0]
print "nClusters: ", clustersN

# Compute covariance Matrix.
vtjcs=tjcs.reshape((tjcsN*tjcsLength, tjcs.shape[2]))
covariance = np.round(np.cov(vtjcs.T)*np.eye(vtjcs.shape[1]))

### Let's iterate with the E-M algorithm ###
maxIter = 100
tol = 0.001
ll_old = 0
for i in xrange(maxIter):
	# E-M algorithm.
    clusters = em.expectation(tjcs, means, covariance, pobty.t_gaussian)
    em.maximization(tjcs, clusters, means, pobty.t_zero, pobty.t_cummulate)
    
    # Update log likelihoood.
    ll_new = 0.0
    for j in range(tjcsN):
		s = 0
		for k in range(clustersN):
			s += pobty.t_gaussian(means[k], covariance, tjcs[j])
		ll_new += np.log(s)
		
    if np.abs(ll_new - ll_old) < tol:
		break
	
    ll_old = ll_new
	    
    print "Iteration Number: ", i 
    
# This is to plot the results of the E-M algorithm.
fig2 = plt.figure()
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("600x500+700+50")
plt.axis([0,16,0,12])
plt.xlabel('X (mts)')
plt.ylabel('Y (mts)')
plt.title("Clustering, Iteration Number: %s" %(i+1))
myplt.plot_clusters(clusters, tjcs)

# Save means structre, which represents the clusters typical trayectories into
# a npy file.
np.save("data/clusters.npy", means)
    
# Let's plot the found clusters.
fig3 = plt.figure()
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("600x500+350+200")
fig3 = plt.title("Clusters")
plt.grid()
plt.xticks(np.arange(0, 16, 1.0))
plt.yticks(np.arange(0, 12, 1.0))
plt.xlabel('X (mts)')
plt.ylabel('Y (mts)')
fig3 = myplt.plot_time_model(means)

# Plot cluster contribs
cluster_contribs = np.sum(clusters, 0)
fig4 = plt.figure()
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("600x500+350+200")
fig4 = plt.title("Number of Trajectories per Cluster")
fig4 = plt.bar(np.arange(clustersN), cluster_contribs, color = myplt.generate_palette(clustersN))
plt.show()
