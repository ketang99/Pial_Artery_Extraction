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
# if you want the graph name in the output, change the below to output_name = f'{graph_name}_restofyourname'
output_name = '' # output name without .vtp
output_igraph_vtp_path = f'{output_dir}/{output_name}'  # desired output vtp path excluding '.vtp'

# the csv file which contains the details about the lobe boundaries
# the directory where it is stored
csv_dir = ''
# the name of the file without .csv
csv_name = ''

graph_name = "BL6_1_VesselVio_clean"
graph_dir = f"{home_dir}/Oct_files"
output_dir = graph_dir
# where the new mesh will be saved
output_igraph_vtp_path = f'{graph_dir}/{graph_name}'

csv_dir = graph_dir
csv_name = f'sample_lobe_line_test_full_BL6_1_OV'


# from the csv, you can select whichever rows/lines you want to be scanned instead of all of them
selected_rows = [] # set this to your desired row indices to be scanned. set to [] if you want all rows to be scanned

sys.path.append(home_dir)
sys.path.append(code_dir)

sys.path.append(home_dir)
sys.path.append(f'{home_dir}/sample_set')

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
parser.add_argument('--neighbourhood', '-n', type=int, metavar='N', default=20)
parser.add_argument('--margin', '-m', type=float, metavar='N', default=10.0)
parser.add_argument('--threshfactor', '-t', type=float, metavar='N', default=5.0)
parser.add_argument('--slicefactor', '-s', type=float, metavar='N', default=2.0)
parser.add_argument('--kdeltafactor', '-k', type=float, metavar='N', default=2.0)


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
    neighbourhood = adj[idcs]
    neigh_distances = traverse_distances[idcs]
    scan_distance_init = traverse_distances[scan_point_id]

    d_min = np.min(neigh_distances)# if face_ascending else np.max(neigh_distances)
    d_max = np.max(neigh_distances)# if face_ascending else np.min(neigh_distances)
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
    half_interval = np.max(bounds[scanplane]) if face_ascending else np.min(bounds[scanplane])
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
    print(f'traverse distances [:15]: {traverse_distances[:15]}')

    print(f'point idcs != -1: {np.count_nonzero(traverse_point_inds >= 0)}')
    # print(f'len of traverse_distances: {len(traverse_distances)}')
    print('Done')

    # Now, we have the min traverse distances for each scan point as well as the point_ind they will hit

    # next step: for each scan point look at the neighbourhood. Run the function to find the ideal traversal distance
    print('Finding ideal traverse distances and updating hull iterating over all scan points')
    # desired_traverse = []
    for i,scan_point in enumerate(grid):
        traverse_good = find_ideal_traversal(scan_point, i, adj, traverse_distances, scanbounds, slice_width, face_ascending, k)
        point_min_traversed = traverse_single_scan_point(points, scan_point, combo, bounds, 
                                                         scanplane, traverse_good, d_thresh, slice_width, face_ascending)
        if len(point_min_traversed) != 0:
            points_hull = np.unique(np.concatenate((points_hull, np.array(point_min_traversed).reshape(-1,3))), axis=0)
            #print(f'Got a new points_hull, {i}, {scan_point}')
            #return points_hull

    print('DONE\n\n')

    return points_hull


def build_grid_strip_along_line_2D(p0, p1, spacing, margin, neighbor_distance, bounds_2d):
    """
    Create a 2D strip of points between p0 and p1, bounded by bounds_2d.

    Parameters:
        p0, p1: 2D endpoints of the scan line
        spacing: distance between points along and across
        margin: half-width of the strip perpendicular to the line
        neighbor_distance: maximum distance between neighbors in adjacency
        bounds_2d: 2x2 array of min/max bounds for the lateral axes, shape (2,2)
    """
    p0 = np.array(p0)
    p1 = np.array(p1)

    # Direction along the line
    line_vec = p1 - p0
    length = np.linalg.norm(line_vec)
    if length == 0:
        raise ValueError("p0 and p1 cannot be the same point.")
    direction = line_vec / length

    # Orthogonal direction (90 degrees)
    orthogonal = np.array([-direction[1], direction[0]])

    n_along = int(np.ceil(length / spacing)) + 1
    n_across = int(np.ceil((2 * margin) / spacing)) + 1

    grid = []
    for i in range(n_along):
        base_point = p0 + i * spacing * direction
        for j in range(n_across):
            offset = (j - n_across // 2) * spacing
            point = base_point + offset * orthogonal

            # Clip to lateral bounds
#            if (bounds_2d[0, 0] <= point[0] <= bounds_2d[0, 1]) and \
#               (bounds_2d[1, 0] <= point[1] <= bounds_2d[1, 1]):
#                grid.append(point)

            grid.append(point)

    grid = np.array(grid)
    n_points = grid.shape[0]

    # Build adjacency matrix
    A = np.zeros((n_points, n_points), dtype=np.uint8)
    for i in range(n_points):
        for j in range(n_points):
            if i != j and np.linalg.norm(grid[i] - grid[j]) <= neighbor_distance:
                A[i, j] = 1

    return {
        'grid': grid,
        'adjacency': A,
        'direction': direction
    }


def get_points_hull_from_row(row, points, spacing, margin, neighbor_distance,
                              slice_factor, thresh_factor, min_dist=2.0, k=2.5):
    """
    Given one row of the DataFrame and a point cloud, return:
    - points_hull: the extracted points
    - num_points: how many points were extracted
    """
    ax_id = row['ax_id']
    combo = [[0,1], [0,2], [1,2]][2-ax_id]
    face_ascending = bool(row['face_ascending'])

    # Define p0 and p1 in 2D
    p0 = np.array([row['p0_coord0'], row['p0_coord1']])
    p1 = np.array([row['p1_coord0'], row['p1_coord1']])

    # All axis bounds (x=0, y=1, z=2)
    all_line_bounds = np.array([
        [row['x_min'], row['x_max']],
        [row['y_min'], row['y_max']],
        [row['z_min'], row['z_max']]
    ])

    # Lateral bounds for grid generation
    bounds_2d = all_line_bounds[combo, :]

    # Axial bounds sorted
    axial_bounds = np.sort([row['axial_bound0'], row['axial_bound1']])

    # Generate grid and adjacency matrix
    result = build_grid_strip_along_line_2D(
        p0=p0,
        p1=p1,
        spacing=spacing,
        margin=margin,
        neighbor_distance=neighbor_distance,
        bounds_2d=bounds_2d
    )

    grid = result['grid']
    adj = result['adjacency']

    print('grid shape: ', grid.shape)

    # Run traversal
    points_hull = all_find_min_traverse(
        points=points,
        points_hull=np.empty((0, 3)),  # start empty
        grid=grid,
        adj=adj,
        combo=combo,
        bounds=all_line_bounds,
        grid_width=spacing,
        slice_factor=slice_factor,
        thresh_factor=thresh_factor,
        min_dist=min_dist,
        k=k,
        face_ascending=face_ascending
    )

    num_points = len(points_hull)
    return points_hull, num_points


def main():

    main_start = time.time()
    global args
    args = parser.parse_args()

    mesh = pv.read(f'{data_dir}/{graph_name}.vtp')
    points = mesh.points
    lines = mesh.lines.reshape(-1, 3)[:, 1:]  # Extract edges (ignore the first count value)
    print(f'lines shape raw = {lines.shape}')
    # get bounds of all the nodes
    bounds_min = np.min(points, axis=0)
    bounds_max = np.max(points, axis=0)

    bounds = np.array([bounds_min, bounds_max])
    bounds = bounds.T

    min_dist = args.dmin
    gridfactor = args.gridfactor
    spacing = min_dist * gridfactor
    neighbourhood = args.neighbourhood
    neighbourhood_grid = int(neighbourhood / spacing)
    thresh_factor = args.threshfactor
    slice_factor = args.slicefactor
    k = args.kdeltafactor
    margin = args.margin

    output_suffix = f'k{k}_neigh{neighbourhood}_thresh{thresh_factor}_spacing{spacing}_slice{slice_factor}_margin{margin}'
    output_hull_path = f'{output_igraph_vtp_path}_lobe_hull_{output_suffix}'

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

    # Read the CSV
    df = pd.read_csv(f'{csv_dir}/{csv_name}.csv')

    # Convert face_ascending from 0/1 or '0'/'1' to boolean
    df['face_ascending'] = df['face_ascending'].astype(str).map({'0': False, '1': True})

    # Ensure the coord columns are floats
    float_cols = ['p0_coord0', 'p0_coord1', 'p1_coord0', 'p1_coord1', 'axial_bound0', 'axial_bound1', 'x_min', 'x_max', 'y_min', 'y_max','z_min', 'z_max']
    df[float_cols] = df[float_cols].astype(float)

    extracted_counts = []
    all_points_hull = []  # Optional: store each points_hull array
    node_labels = np.zeros(len(points))

    print('Lobe boundary csv has been read, beginning scan:\n\n')

    for idx, row in df.iterrows():
        print(f"Processing line index {idx}...")
        iter_time = time.time()

        if idx not in selected_rows and len(selected_rows)!=0:
            continue
        elif not selected_rows or idx in selected_rows:
            points_hull, num_points = get_points_hull_from_row(
                row=row,
                points=points,
                spacing=spacing,
                margin=margin,
                neighbor_distance=neighbourhood,
                slice_factor=slice_factor,
                thresh_factor=thresh_factor,
                min_dist=min_dist, 
                k=k
            )
            
            # return
            # extracted_counts.append(num_points)
            all_points_hull.append(points_hull)

            hull_idcs = []
            for p in points_hull:
                bool_check = [points[:,0]==p[0], points[:,1]==p[1], points[:,2]==p[2]]
                points_match = np.all(bool_check, axis=0)
                hull_idcs.append(np.where(points_match==True))

            hull_idcs = np.array(hull_idcs).astype(int)
            hull_idcs = hull_idcs.reshape((hull_idcs.shape[0],))

            node_labels[hull_idcs] = 1
            mesh.point_data['node_label'] = node_labels
            mesh.save(f'{output_hull_path}_lineID{idx}.vtp')
            print('\nhull vtp saved')
            print(f'Iteration runtime = {(time.time() - iter_time)/60.0} mins')

            counts = idx * np.ones(num_points)
            extracted_counts.append(counts)
            print(f'number of surface points found for iteration = {extracted_counts[-1]}\n*****\n')

    points_hull = np.vstack(all_points_hull)
    extracted_counts = np.concatenate(extracted_counts)
    
    print(f'points hull shape: {points_hull.shape}')
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

    hull_idcs = np.array(hull_idcs).astype(int)
    hull_idcs = hull_idcs.reshape((hull_idcs.shape[0],))

    node_labels = np.zeros(len(points))
    node_labels[hull_idcs] = 1
    mesh.point_data['node_label'] = node_labels
    mesh.save(f'{output_hull_path}.vtp')
    print('hull vtp saved')
    print('\n*****\n SCRIPT COMPLETE\n*****')
    
    # # first select lines that contain just the hull idcs
    # lines_hull = []
    # for l in lines:
    #     if l[0] in hull_idcs and l[1] in hull_idcs:
    #         lines_hull.append(l)

    # lines_hull = np.array(lines_hull)
    
    # print(f'hull_idcs shape: {hull_idcs.shape}')
    # print(f'lines_hull shape: {lines_hull.shape}')

    # edges_thresh_c = np.copy(lines_hull)
    # for i,val in enumerate(hull_idcs):
    #     edges_thresh_c[np.where(lines_hull==val)] = i

    # graph_hull = ig.Graph()
    # graph_hull.add_vertices(len(points_hull))
    # graph_hull.vs['coords'] = points_hull
    # graph_hull.add_edges(edges_thresh_c)

    # write_vtp(graph_hull, f'{output_hull_path}.vtp', coordinatesKey='coords')
    # print('hull vtp saved')


if __name__ == '__main__':
    main()