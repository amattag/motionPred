"""
Clustering with the Expectation-Maximization (E-M) algorithm

Demonstration of how to cluster a set of trajectories using the 
Expectation-Maximization (E-M) algorithm.
"""

# Author: Antonio Matta <antonio.matta@upm.es>

# Import required modules.
import numpy as np
import matplotlib.pyplot as plt
import random

# import auxiliar modules.
import expMax as em
import plotting as myplt
import probability as pobty

# Load trayectory data from the 'tjcs.npy' file.
# A set of 1856 trajectories generated from the Edinburgh Informatics Forum 
# Pedestrian Database: http://homepages.inf.ed.ac.uk/rbf/FORUMTRACKING/
data = np.load("data/tjcs.npy")

# Choose the first 500 trajectories.
tjcs = data[:500]

# Plot selected trajectories
#wm = plt.get_current_fig_manager()
#wm.window.wm_geometry("800x900+50+50")
fig1 = plt.figure()
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("600x500+50+50")
fig1 = plt.title("Trajectories")
plt.axis([0,16,0,12])
plt.xlabel('X (mts)')
plt.ylabel('Y (mts)')
myplt.plot_trajectories(tjcs)

# Initialize the initial mean values structure.
clustersN = 10                # Number of Clusters (it must be from console).
tjcsN = tjcs.shape[0]         # Number of Trajectories. 
tjcsLength = tjcs.shape[1]    # Lenght of each trajectory.
means = np.zeros((clustersN, tjcsLength, 2))

# Fill initial means structure by selecting trajectories at random.
tjcIndex = random.sample(np.arange(tjcsN), clustersN)
for m, n in zip(xrange(clustersN), tjcIndex):
    means[m] = tjcs[n]
   
# Compute covariance Matrix.
vtjcs=tjcs.reshape((tjcsN*tjcsLength, tjcs.shape[2]))
covariance = np.round(np.cov(vtjcs.T)*np.eye(vtjcs.shape[1]))

### Let's iterate with the E-M algorithm applying an heuristic that optimizes 
# the number of trajectories on every cluster ###

# This is to plot the results of the E-M algorithm.
plt.pause(0.05)
fig2 = plt.figure()
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("600x500+700+50")
plt.axis([0,16,0,12])
plt.xlabel('X (mts)')
plt.ylabel('Y (mts)')
plt.ion()

iterations = 20
for i in xrange(iterations):
    clusters = em.expectation(tjcs, means, covariance, pobty.t_gaussian)
    em.maximization(tjcs, clusters, means, pobty.t_zero, pobty.t_cummulate)
    # Each iteration, The E-M algorithm is run five times.
    #for j in xrange(5):
        #clusters = em.expectation(tjcs, means, covariance, pobty.t_gaussian)
        #em.maximization(tjcs, clusters, means, pobty.t_zero, pobty.t_cummulate)
    
    # Replace the worst cluster with the worst represented trajectory 
    # to improve the quality of the clusters.    
    if i < iterations - 1:
        c_index, c_score = em.worst_cluster(clusters)
        t_index, t_score = em.worst_trajectory(clusters, c_index, c_score, 
                                               tjcIndex, tjcs, covariance)
        # If a worst trajectory is not found.
        if t_index == -1:
            #break
            means[c_index] = -1
        else:    
            #print "Replacing cluster %i with trajectory %i" % ( c_index, t_index )
            means[c_index] = tjcs[t_index]
            tjcIndex.append( t_index )
            
    # This is for plotting the results.
    plt.title("Clustering, Iteration Number: %s" %(i+1))
    myplt.plot_clusters(clusters, tjcs)
    plt.pause(0.01)

# Save means structre, which represents the clusters typical trayectories into
# a npy file.
np.save("data/clusters.npy", means)
    
# Let's plot the found clusters.
fig3 = plt.figure()
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("600x500+350+200")
fi3 = plt.title("Clusters")
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

while True:
    plt.pause(0.01)
