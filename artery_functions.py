'''
Module that contains functions for brain artery graph manipulation and analysis
'''

import numpy as np
import pandas as pd
import igraph as ig
import pyvista as pv
from scipy.spatial import KDTree
import os


'''
The following 2 functions take a pyvista mesh object and return:
- edges_all: all the edges in the form [node1, node2] and shape [E,2] where E=number of edges
- edges_thresh: the edges that only include nodes that satisfy the cortical depth threshold inputted
- points_all: all the coordinates of the nodes
- points_idcs_thresh: the indices of the nodes that fall under the cortical depth threshold

mesh: the pyvista mesh
'''
def obtain_edges(mesh):

  edges = mesh.lines.reshape(-1,3)[:,1:]

  return edges # edges returned as a [E,2] array

'''
Returns edges of nodes that lie below a threshold for cortical depth
'''
def obtain_edges_thresholded(mesh, thresh):

  points_all = mesh.points
  depths = mesh.point_data['cortical_depth_from_outer_wall']
  points_idcs_thresh = np.where(depths<=thresh)

  edges_all = obtain_edges(mesh)
  edges_remove = []
  for i in range(len(edges_all)):
    id0 = edges_all[i][0]
    id1 = edges_all[i][1]
    if not (depths[id0] <= thresh and depths[id1] <= thresh):
      edges_remove.append(i)

  edges_thresh = np.delete(edges_all, edges_remove, axis=0)

  return edges_all, edges_thresh, points_all, np.array(points_idcs_thresh[0])

'''
Get the indices of hull points with respect to the complete brain graph nodes
'''
def get_hull_idcs(points_hull, points_all):
  hull_idcs = []
  for p in points_hull:
      bool_check = [points_all[:,0]==p[0], points_all[:,1]==p[1], points_all[:,2]==p[2]]
      points_match = np.all(bool_check, axis=0)
      hull_idcs.append(np.where(points_match==True))

  hull_idcs = np.array(hull_idcs)
  hull_idcs = hull_idcs.reshape((hull_idcs.shape[0],))

  return hull_idcs

'''
Read the convex hull points from the hull csv and get their indices
'''
def get_hull(read_csv_path, points_all):

  df_hull = pd.read_csv(read_csv_path)
  points_hull = df_hull[['X','Y','Z']].to_numpy()

  hull_idcs = get_hull_idcs(points_hull, points_all)

  return points_hull, hull_idcs

'''
Find the neighboring nodes of the input node_id as well as the degree of that node
edges is of shape [E,2] where E is number of edges and each column has node IDs that are connected
'''
def get_neighbours(node_id, edges):

  edges_id = np.where(edges==node_id)
  edges_id = edges[edges_id[0]]

  neighbors_id = np.unique(edges_id)
  neighbors_id = np.delete(neighbors_id, neighbors_id==node_id)

  return neighbors_id, len(neighbors_id)


'''
Assign any thresholded nodes with a degree of >=3 to the hull
'''
def add_deg3_hull(points_idcs_thresh, hull_idcs, edges_thresh):
    
    degs_thresh = []
    new_hull_idcs = []
    for n in points_idcs_thresh:
      if n not in hull_idcs:
        _, deg = get_neighbours(n, edges_thresh)
        if deg >= 3:
          degs_thresh.append(deg)
          new_hull_idcs.append(n)
                
    return new_hull_idcs, degs_thresh

'''
for any points which are connected to >=2 points on the hull, add these to the hull
'''
def connect_hull_points(points_idcs_thresh, hull_idcs, edges_thresh):

  degs_thresh_e = []
  new_hull_idcs_e = []
  for n in points_idcs_thresh:
    # print(n)
    if n not in hull_idcs:
      neighs, deg = get_neighbours(n, edges_thresh)
      if deg >= 2:
        neighhullcount = 0
        for nn in neighs:
          if nn in hull_idcs:
            neighhullcount += 1
        if neighhullcount >= 2:
          new_hull_idcs_e.append(int(n))

  return new_hull_idcs_e, degs_thresh_e