"""
Plotting module

A module with auxiliary functions to plot clusters and trajectories.
"""

# Author: Antonio Matta <antonio.matta@upm.es>

# Import required modules.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

palettes = {}

def generate_palette( count ):
  """Generates a color id to every cluster.
  
  Parameters
  ----------
  count: int
    Number of clusters or trajectories.
  
  Returns
  -------
  palettes: array_like
    An array of colored ids. One for each cluster.
  """  
  if count in palettes:
    return palettes[count]
  samples = np.random.randint( 0, 255, ( 1000, 3 ) ) 
  km = KMeans( count, init = "k-means++" )
  km.fit( samples )
  palettes[count] = km.cluster_centers_ / 255.0
  return palettes[count]


def plot_trajectories( data ):
  """Plots a set of trajectories.
  
  Parameters
  ----------
  data: array 
    A set of 2D (x, y) trajectories. Each point of is a float number.
  """ 
  if len( data ) < 50:
    palette = generate_palette( len( data ) )
    for n, t in enumerate( data ):
      t = np.array( t )
      plt.plot( t[:,0], t[:,1], "-", c = palette[n] )
      plt.plot( t[-1,0], t[-1,1], "o", markersize = 5.0, c = palette[n] )
  else:
    for t in data:
      t = np.array( t )
      plt.plot( t[:,0], t[:,1], "-" )


def plot_clusters( e, data ):
  """Plots all the clusters and their corresponding trajectories.
  
  Parameters
  ----------
  e: array_like
    A set of clusters. Each point of is a float number.
  
  data: array_like
    A set of 2D (x, y) trajectories. Each point of is a float number.
  """ 
  M = e.shape[1]
  palette = generate_palette( M )
  e_total = np.sum( e )
  for n, traj in enumerate( data ):
    traj = np.array( traj )
    c = np.array( [0., 0., 0.] )
    if np.sum( e[n, :] ) / e_total > 1E-4:
      e_m = np.sum( e[n, :] )
      for m in xrange( M ):
        c += palette[m] * e[n, m] / e_m
      plt.plot( traj[:,0], traj[:,1], "-", color = c )
    else:
      c = np.array( [0.8, 0.8, 0.8] )
      plt.plot( traj[:,0], traj[:,1], "-", color = c, alpha = 0.3 )
        
def plot_time_model( data ):
  """From every cluster, plot a typical trajectory that represents the 
     entire cluster.
  
  Parameters
  ----------
  data: array
      An array containing all the clusters.
  """   
  palette = generate_palette(len(data))
  for n, t in enumerate(data):
    t = np.array(t)
    plt.plot(t[:,0], t[:,1], "o", c = palette[n])


def plot_prediction( clusters, obs, belief, prediction ):
  """ Plot the observations, current belief state and predicted belief state at
  a time step given by the user.
  
  Parameters
  ----------
  clusters: array.
    An array containing all the clusters.
      
  obs: array.
    Set of trajectories used as observations..  
  
  belief: matrix array
    State at time t
    
  prediction: matrix array
    State at time t+steps
  """ 
  palette = generate_palette(len(clusters))
  
  # Plot the observations (trajectories)
  plt.plot(obs[:,0], obs[:,1], "x", c = (0.8, 0.1, 0.3))
  
  for n, t in enumerate(clusters):
    t = np.array(t)
    for i in xrange(len(t)):
      # Plot the belief state at the current time step.  
      plt.plot(t[i,0], t[i,1], "o", markersize = 15. * belief[n, i], c = (0.2, 0.6, 0.2))
      
      # Plot the belief state in a future time step. 
      plt.plot(t[i,0], t[i,1], "D", markersize = 60. * prediction[n, i], c = (1.0, 0.3, 0.3))
