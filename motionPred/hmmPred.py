"""
Human motion prediction using a HMM

Demonstration of how to predict human motion using a Hidden Markov Model.
"""

# Author: Antonio Matta <antonio.matta@upm.es>

# Import required modules.
import numpy as np
import matplotlib.pyplot as plt
import random

# import auxiliar modules.
import plotting as myplt

# Load clusters trayectory data from the 'clusters.npy' file.
clusters = np.load("data/clusters.npy")

# Let's plot those clusters again.
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("700x600+300+70")
plt.title("Clusters")
plt.grid()
plt.xticks(np.arange(0, 16, 1.0))
plt.yticks(np.arange(0, 12, 1.0))
plt.xlabel('X (mts)')
plt.ylabel('Y (mts)')
myplt.plot_time_model(clusters)
plt.show()

# Resample the original clusters every two samples (0.6 seconds).
clustersN, origSteps, dim = clusters.shape
stepsN = int(np.ceil(origSteps/3.0))
means = np.zeros((clustersN, stepsN, dim))
for m in xrange(clustersN):
    for t in xrange(stepsN):
        count = 0.0
        for new_t in xrange(3 * t, min(3 *(t + 1), origSteps)):
            means[m, t] += clusters[m, new_t]
            count += 1.0
        means[m, t] /= count

# Plot the new clusters
fig2 = plt.figure()
wm = plt.get_current_fig_manager()
wm.window.wm_geometry("700x600+300+70")
fig2 = plt.title("Clusters")
plt.grid()
plt.xticks(np.arange(0, 16, 1.0))
plt.yticks(np.arange(0, 12, 1.0))
plt.xlabel('X (mts)')
plt.ylabel('Y (mts)')
myplt.plot_time_model(means)
plt.show()

### Let us create the HMM model. ###

# First step: The Transition Probability Matrix A
A = np.zeros(( clustersN, stepsN, stepsN ))
for m in xrange(clustersN):
    for t in xrange(stepsN):
        if t < stepsN - 1:
            # Normal node
            A[m, t, t + 1] = 1. / 3. # Advance
            A[m, t, t] = 2. / 3.     # Stay in node
        else:
            # End node
            A[m, t, t] = 1.

# Second step: The state prior.
# A motion always starts at the beginning of the trajectory.
pi = np.zeros((clustersN, stepsN))
for m in xrange(clustersN):
    pi[m, 0] = 1.0
pi /= np.sum(pi)

# Third step: The global covariance matrix for the observation probabilities.
covariance = np.array( [[4.0, 0.0], [0.0, 4.0]] )
cov_inv = np.linalg.inv( covariance )

### Using the HMM model for prediction. ###

# The Filtering algorithm.
def filter(belief, A, means, cov_inv, obs, obsStep):
    """filter: To estimate the probability distribution over the state at the
       current time.
    
    Parameters
    ----------
    belief: array
      Initial belief (state prior).
    
    A: array
      Transition Probability Matrix.
      
    means: array
      Clusters array.
    
    cov_inv: array matrix
      Inverse of the Covariance Matrix
    
    obs: array
      Trajectory array used as an observation.
      
    step: int
      State step of the coming observation.
    """
    #clustersN, new_t = belief.shape
    #current = np.zeros((clustersN, new_t))
    #for m in xrange(clustersN):
        #for i in xrange(new_t):
            #for old_t in xrange(new_t):
                #diff = means[m, i] - obs[step]
                #expo = -0.5 * np.dot(np.dot(diff, cov_inv), diff.transpose())
                #current[m, i] += belief[m, old_t] * A[m, old_t, i] * np.exp(expo) 
    #current /= np.sum(current)
    #belief[:] = current[:]
    
    clustersN, stepsN = belief.shape
    currentState = np.zeros((clustersN, stepsN))
    for m in xrange(clustersN):
        for currentStep in xrange(stepsN):
			diff = means[m, currentStep] - obs[obsStep]
			expo = -0.5 * np.dot(np.dot(diff, cov_inv), diff.transpose())
			for prevStep in xrange(stepsN):
				currentState[m, currentStep] += belief[m, prevStep] * A[m, prevStep, currentStep] * np.exp(expo) 
    currentState /= np.sum(currentState)
    belief[:] = currentState[:]
    
# The Prediction algorithm.
def predict(belief, A, stepsAhead):
    """predict: To estimate the probability distribution at different time steps
    in the future. 
    
    Parameters
    ----------
    belief: array
      State belief at time t.
    
    A: array
      Transition Probability Matrix.
      
    stepsAhead: int
      Number of states ahead in the future to make a prediction.
     
    Returns
    -------
    belief: array matrix
      Prediction of the State at a number of steps in the future.
    """
    clustersN, stepsN = belief.shape
    for t in xrange(stepsAhead):
        currentState = np.zeros((clustersN, stepsN))
        for m in xrange(clustersN):
            for currentStep in xrange(stepsN):
                for prevStep in xrange(stepsN):
                    currentState[m, currentStep] += belief[m, prevStep] * A[m, prevStep, currentStep]
        currentState /= np.sum(currentState)
        belief = currentState
    return belief
    
### Test the filtering and prediction algorithms with some test data. ###

# Grab the last 100 trajectories as test data.
data = np.load("data/tjcs.npy")
test_data = data[-100:]

iniState = 0    # initial State
endState = stepsN   # final State
interval = 2    # State Step interval
stepsAhead = 10 # Number of steps ahead in the future

for obs in test_data[34:37]:  # Use tjcs 34 to 36 as observations. 
    belief = pi.copy()
    for obsStep in xrange(iniState, (endState+1), interval):
        filter(belief, A, means, cov_inv, obs, obsStep)
        prediction = predict(belief, A, stepsAhead)
        
        #Plot prediction
        wm = plt.get_current_fig_manager()
        wm.window.wm_geometry("700x600+300+70")
        plt.grid()
        plt.xticks(np.arange(0, 16, 1.0))
        plt.yticks(np.arange(0, 12, 1.0))
        plt.xlabel('X (mts)')
        plt.ylabel('Y (mts)')  
        plt.title("Prediction. State Step: %s" %obsStep)
        myplt.plot_prediction(means, obs[:obsStep], belief, prediction)        
        plt.show()
