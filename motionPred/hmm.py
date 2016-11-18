"""
Human motion prediction using a HMM

Hidden Markov Model model.
"""

# Author: Antonio Matta <antonio.matta@upm.es>

# Import required modules.
import numpy as np

# The Transition Probability Matrix A
def transProbMat(clustersN, stepsN):
	"""
	"""
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
	return A

# State Prior
def statePrior(clustersN, stepsN):
	"""
	"""
	# A motion always starts at the beginning of the trajectory.
	pi = np.zeros((clustersN, stepsN))
	for m in xrange(clustersN):
		pi[m, 0] = 1.0
	pi /= np.sum(pi)
	return pi


# The Filtering algorithm.
def filter(belief, A, means, invCov, obs, obsStep):
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
      
    obsStep: int
      State step of the coming observation.
    """
    clustersN, stepsN = belief.shape
    currentState = np.zeros((clustersN, stepsN))
    for m in xrange(clustersN):
        for currentStep in xrange(stepsN):
			diff = means[m, currentStep] - obs[obsStep]
			expo = -0.5 * np.dot(np.dot(diff, invCov), diff.transpose())
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
