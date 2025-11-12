import torch
torch.cuda.is_available()
import sys

# directory that contains the py files
code_dir = '' 
sys.path.append(code_dir)

# home directory which has datasets, codes etc
home_dir = '' 

# full data directory that contains the vtp file to be scanned
data_dir = ''
# name of the vtp file
graph_name = ''

# desired output directory. Is set to data_dir currently
output_dir = data_dir

# desired output name. Script will save the result to data_dir
output_name = '' 
output_igraph_vtp_path = f'{output_dir}/{output_name}'  # desired output vtp path excluding '.vtp'

planes_reject = []

sys.path.append(home_dir)
sys.path.append(code_dir)

import numpy as np
import pandas as pd
import igraph as ig
import pyvista as pv
import argparse
import time

from write_vtp_py3 import write_vtp

from utils import *

parser = argparse.ArgumentParser(description='Hull stopping extractor')
parser.add_argument('--gridfactor', '-g', type=float, metavar='N', default=1)
parser.add_argument('--dmin', '-d', type=float, metavar='N', default=2.0)
parser.add_argument('--neighbourhood', '-n', type=int, metavar='N', default=15)
parser.add_argument('--threshfactor', '-t', type=float, metavar='N', default=5.0)
parser.add_argument('--slicefactor', '-s', type=float, metavar='N', default=2.0)
parser.add_argument('--kdeltafactor', '-k', type=float, metavar='N', default=0.75)


def mesh_adjacency_matrix(rows, cols, N):
    n_points = rows * cols  # Total number of points
    # A = scipy.sparse.lil_matrix((n_points, n_points), dtype=int)  # Sparse adjacency matrix: is very slow
    A = np.zeros((n_points, n_points), dtype=np.uint8)
    print('initialized A')

    for r in range(rows):
      # print(f'r = {r}')
        for c in range(cols):
            index = r * cols + c  # Convert 2D index to 1D

            # Define all neighbors within an N-radius square
            for dr in range(-N, N + 1):
                for dc in range(-N, N + 1):
                    nr, nc = r + dr, c + dc  # Compute neighbor coordinates
                    if 0 <= nr < rows and 0 <= nc < cols:  # Check bounds
                        neighbor_index = nr * cols + nc
                        A[index, neighbor_index] = 1  # Mark as connected

    # return A.tocsr()  # Convert to efficient sparse format
    return A

def post_adjacency(bounds, combo, spacing, neighbourhood):

    # make meshgrid using bounds and mindist
    print(f'bounds[combo]: {bounds[combo]}')
    stepping = []
    for b in bounds[combo]:
        stepping.append(np.arange(b[0],b[1], step=spacing))

    mgrid0, mgrid1 = np.meshgrid(stepping[0], stepping[1])

    # convert meshgrid to N,2 where N is the number of grid points
    coords = np.column_stack((mgrid0.ravel(), mgrid1.ravel()))

    print('Found bounds and coords')

    rows, cols = mgrid0.shape
    print('rows and cols: ', rows, cols)

    adj = mesh_adjacency_matrix(rows, cols, neighbourhood)
    print('Found adjacency matrix')

    return rows, cols, adj, coords

'''
This function will, for a single scan point, find the min distance it must traverse in to find a node

combo is the particular combo of cross section e.g. [0,1] for scanning along +z
scanbounds is either [min, middle] or [max, middle] depending on what face ascending is
'''
def single_find_min_traverse(points, scan_point, combo, scanplane, scanbounds, grid_width, face_ascending):

    grid_interval = grid_width / 2

    # select points
    if face_ascending:
        within_bounds = np.all(np.stack([np.abs(points[:,combo[0]] - scan_point[0]) <= grid_interval,
                                         np.abs(points[:,combo[1]] - scan_point[1]) <= grid_interval,
                                         points[:,scanplane] <= scanbounds[1]]),
                               axis=0)
    else:
        within_bounds = np.all(np.stack([np.abs(points[:,combo[0]] - scan_point[0]) <= grid_interval,
                                         np.abs(points[:,combo[1]] - scan_point[1]) <= grid_interval,
                                         points[:,scanplane] >= scanbounds[1]]),
                               axis=0)

    # print('within bounds shape, ', within_bounds.shape)
    points_selected = points[within_bounds]

    # return

    if len(points_selected) != 0:
        if face_ascending:
            # min_traverse = abs(scanbounds[0] - np.min(points_selected[:,scanplane]))
            min_traverse = np.min(points_selected[:,scanplane])
            ind_chosen = np.argmin(points_selected[:,scanplane])
        else:
            # min_traverse = abs(scanbounds[0] - np.max(points_selected[:,scanplane]))
            min_traverse = np.max(points_selected[:,scanplane])
            ind_chosen = np.argmax(points_selected[:,scanplane])

        point_idx = int(np.arange(len(points))[within_bounds][ind_chosen])

        return float(min_traverse), point_idx

    else:
        return float(scanbounds[1]), -1
    

'''
scanbounds: is either [min,half] or [max, half] for face_ascending true or false respectively
'''

def find_ideal_traversal(scan_point, scan_point_id, adj, traverse_distances, scanbounds, slice_width, face_ascending=True, k=10):
   
    idcs = np.where(adj[scan_point_id]!=0)[0]
    neigh_distances = np.sort(traverse_distances[idcs])
    neighbourhood_size = len(neigh_distances)

    d_min = np.min(neigh_distances)# if face_ascending else np.max(neigh_distances)
    d_max = neigh_distances[-int(np.round(0.1*neighbourhood_size))]
    # d_max = np.max(neigh_distances)# if face_ascending else np.min(neigh_distances)
    scan_b_diff = abs(scanbounds[1] - scanbounds[0])

    if d_min != d_max:
        delta = k * scan_b_diff / (d_max - d_min)

        if face_ascending:
            traverse_start = d_min
            traverse_end = d_min + delta*slice_width
            traverse_end = scanbounds[1] if traverse_end > scanbounds[1] else traverse_end
        else:
            traverse_end = d_max
            traverse_start = d_max - delta*slice_width
            traverse_start = scanbounds[1] if traverse_start < scanbounds[1] else traverse_start
    
    else:
        traverse_start = d_min
        traverse_end = d_min

    return [traverse_start, traverse_end]


'''
Traverse bounds must be given in the form [tmin, tmax] regardless of face_ascending's value
'''

def traverse_single_scan_point(points, scan_point, combo, bounds, scanplane, traverse_good, d_thresh, slice_width, face_ascending=True):

    # determine the bounds of our axial window (start-slice_width to end+slice_width)
    traverse_bounds = []
    traverse_bounds.append(float(traverse_good[0]))
    traverse_bounds.append(float(traverse_good[1]))

    # select points that lie within the axial window and are in a square radius <= d_thresh
    within_bounds = np.all(np.stack([np.abs(points[:,combo[0]] - scan_point[0]) < d_thresh,
                                        np.abs(points[:,combo[1]] - scan_point[1]) < d_thresh,
                                        points[:,scanplane] >= traverse_bounds[0],
                                        points[:,scanplane] <= traverse_bounds[1]]),
                            axis=0)

    points_selected = points[within_bounds]

    # select the point which has the minimum traveling distance

    ## To potentially do: if a point has already been put in hull then go to the next one

    if len(points_selected != 0):
        point_min_traversed = points_selected[np.argmin(points_selected[:,scanplane])] if face_ascending else points_selected[np.argmax(points_selected[:,scanplane])]
        
    else:
        point_min_traversed = []

    # *** THIS IS THE SAME AS SINGLE FIND MIN TRAVERSE SO JUST ADJUST THAT A BIT  ***

    return point_min_traversed


'''
This function will get the minimum traverse distance for all scan points and store them in an array

scanbounds: is either [min,half] or [max, half] for face_ascending true or false respectively
'''
def all_find_min_traverse(points, points_hull, grid, adj, combo, bounds, grid_width, slice_factor,
                          thresh_factor, min_dist=2.0, k=2.5, face_ascending=True):

    slice_width = slice_factor * min_dist
    d_thresh = thresh_factor * min_dist

    for c in range(3):
        if c not in combo:
            scanplane = c

    half_interval = np.mean(bounds[scanplane])
    scanbounds = [bounds[scanplane][0], half_interval] if face_ascending else [bounds[scanplane][1], half_interval]

    print('\n\nFinding initial traverse distances for all scan points')
    traverse_distances = []
    traverse_point_inds = []
    for scan_point in grid:
        min_traverse, min_ind = single_find_min_traverse(points, scan_point, combo, scanplane, scanbounds, grid_width, face_ascending)
        traverse_distances.append(min_traverse)
        traverse_point_inds.append(min_ind)
    
    traverse_distances = np.array(traverse_distances)
    print(f'traverse distances shape: {traverse_distances.shape}')
    traverse_point_inds = np.array(traverse_point_inds)
    
    print(f'point idcs != -1: {np.count_nonzero(traverse_point_inds >= 0)}')
    # print(f'len of traverse_distances: {len(traverse_distances)}')
    print('Done')

    # Now, we have the min traverse distances for each scan point as well as the point_ind they will hit

    print('Finding ideal traverse distances and updating hull iterating over all scan points')
    # desired_traverse = []
    for i,scan_point in enumerate(grid):
        # find the ideal traversal axial window for a scan point
        traverse_good = find_ideal_traversal(scan_point, i, adj, traverse_distances, scanbounds, slice_width, face_ascending, k)
        # search within this window to find a suitable surface point to add to the hull
        point_min_traversed = traverse_single_scan_point(points, scan_point, combo, bounds, 
                                                         scanplane, traverse_good, d_thresh, slice_width, face_ascending)
        if len(point_min_traversed) != 0:
            points_hull = np.unique(np.concatenate((points_hull, np.array(point_min_traversed).reshape(-1,3))), axis=0)

    print('DONE\n\n')

    return points_hull


def main():

    main_start = time.time()
    global args
    args = parser.parse_args()

    mesh = pv.read(output_igraph_vtp_path)
    points = mesh.points
    lines = mesh.lines.reshape(-1, 3)[:, 1:]  # Extract edges (ignore the first count value)
    print(f'lines shape raw = {lines.shape}')
    # get bounds of all the nodes
    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)

    bounds = np.array([bounds_min, bounds_max])
    bounds = bounds.T

    combos = [[0,1],[0,2],[1,2]]

    min_dist = args.dmin
    gridfactor = args.gridfactor
    spacing = min_dist * gridfactor
    neighbourhood = args.neighbourhood
    neighbourhood_grid = int(neighbourhood / spacing)
    thresh_factor = args.threshfactor
    slice_factor = args.slicefactor
    k = args.kdeltafactor

    output_suffix = f'k{k}_dmin{min_dist}_neigh{neighbourhood}_thresh{thresh_factor}_spacing{spacing}_slice{slice_factor}'
    output_hull_path = f'{output_igraph_vtp_path}_hull_{output_suffix}'
    output_all_path = f'{output_igraph_vtp_path}_all_{output_suffix}'
    
    print(f'min_dist/um = {min_dist}')
    print(f'grid spacing factor = {gridfactor}')
    print(f'spacing/um = {spacing}')
    print(f'neighbourhood radius = {neighbourhood}')
    print(f'neighbourhood grid squares = {neighbourhood_grid}')
    print(f'thresh_factor distance = {thresh_factor}')
    print(f'slice_factor = {slice_factor}')
    print(f'output hull path: {output_hull_path}\n')
    
    print(f'bounds = {bounds}')
    
    print(f'k = {k}\n\n')

    print('GETTING POINTS HULL')
    points_hull = np.zeros((1,3))

    for cid in range(3):
        if (cid, True) in planes_reject and (cid, False) in planes_reject:
            continue
        elif (cid, True) in planes_reject:
            faces = [False]
        elif (cid, False) in planes_reject:
            faces = [True]
        else:
            faces = [True, False]

        combo = combos[cid]
        print(f'combo id: {cid}, combo={combo}')
        print('Finding adjacency matrix')

        _,_, adj, grid = post_adjacency(bounds, combo, spacing, neighbourhood_grid)
        print('Done\n')

        for face_ascending in faces:
            points_hull = all_find_min_traverse(points, points_hull, grid, adj, combo, bounds, spacing, slice_factor,
                                                thresh_factor, min_dist, k, face_ascending)
   
        print('\nall find min traverse run successfully!')
    
    print('For loop complete')
    print(f'points_hull shape: {points_hull.shape}')
    points_hull = points_hull[~np.all(points_hull == 0, axis=1)]
    print('\n*****\nGot points_hull!\n*****\n')
    
    print(f'points_hull shape: {points_hull.shape}')

    df_hull = pd.DataFrame()
    df_hull['X'] = points_hull[:,0]
    df_hull['Y'] = points_hull[:,1]
    df_hull['Z'] = points_hull[:,2]

    # save csv of the hull points to visualize via table to points on paraview
    df_hull.to_csv(f'{output_hull_path}.csv')
    print('hull csv saved\n')

    # get hull_idcs from points_hull
    hull_idcs = []
    for p in points_hull:
        bool_check = [points[:,0]==p[0], points[:,1]==p[1], points[:,2]==p[2]]
        points_match = np.all(bool_check, axis=0)
        hull_idcs.append(np.where(points_match==True))

    hull_idcs = np.array(hull_idcs)
    hull_idcs = hull_idcs.reshape((hull_idcs.shape[0],)) 
    
    # first select lines that contain just the hull idcs
    lines_hull = []
    for l in lines:
        if l[0] in hull_idcs and l[1] in hull_idcs:
            lines_hull.append(l)

    lines_hull = np.array(lines_hull)
    
    print(f'hull_idcs shape: {hull_idcs.shape}')
    print(f'lines_hull shape: {lines_hull.shape}')

    edges_thresh_c = np.copy(lines_hull)
    for i,val in enumerate(hull_idcs):
        edges_thresh_c[np.where(lines_hull==val)] = i

    graph_hull = ig.Graph()
    graph_hull.add_vertices(len(points_hull))
    graph_hull.vs['coords'] = points_hull
    graph_hull.add_edges(edges_thresh_c)

    # save vtp of just the hull i.e. surface nodes
    write_vtp(graph_hull, f'{output_hull_path}.vtp', coordinatesKey='coords')
    print('hull vtp saved')
    
    # save vtp of the entire graph, but with the surface nodes labeled under 'node_labels'
    node_labels = np.zeros(mesh.n_points)
    node_labels[hull_idcs] = 1
    mesh.point_data['node_label'] = node_labels
    mesh.save(f'{output_all_path}.vtp')

    # some extra code that should not be required. It is an alternate way to store the hull vtp
    # poly = pv.PolyData(points_hull)

    # Convert edges to VTK line format
    # Each line needs to start with the number of points (2), followed by the point indices
    # lines = np.hstack([[2, edge[0], edge[1]] for edge in edges_thresh_c])

    # Add the lines to the PolyData
    # poly.lines = lines

    # poly.save(f'{output_hull_path}.vtp')

    print('complete vtp saved\n')

    print(f'*********\nSCRIPT COMPLETE. Runtime = {(time.time() - main_start)/60.0} mins\n*********')

if __name__ == '__main__':
    main()
