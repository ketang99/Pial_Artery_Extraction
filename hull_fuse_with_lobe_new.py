import sys

# directory that contains the py files
code_dir = '' 
sys.path.append(code_dir)

# home directory which has datasets, codes etc
home_dir = '' 

# full data directory that contains the vtp file to be scanned
data_dir = ''

# name of the vtp files without extension
# graph_name_all should be from hull_tweak_traversal_single.py and have 'all' in its name 
# this is because this has all the points and edges for the complete graph

# each entry should correspond to the X, Y or Z files. Exclude '.vtp' from the names
# if there is just one allnodes file, then just put that filename as a string inside
# the array: should look like ['filename_allnodes']
graph_name_all = ['filenameX',
                  'filenameY',
                  'filenameZ']

# filename for the lobe scan
graph_name_lobe = ''

# desired output directory. Is set to data_dir currently
output_dir = data_dir
# desired output name. Script will save the result to data_dir
# if you want the graph name in the output, change the below line to output_name = f'{graph_name}_restofyourname'
output_name = '' 
output_igraph_vtp_path = f'{output_dir}/{output_name}'  # desired output vtp path excluding '.vtp'

sys.path.append(home_dir)
sys.path.append(code_dir)

import numpy as np
import pandas as pd
import igraph as ig
import pyvista as pv
from scipy.spatial import KDTree

import artery_functions as art

def main():

    all_node_labels = []
    for i,gname in enumerate(graph_name_all):
        mesh_full = pv.read(f'{data_dir}/{gname}.vtp')
        all_node_labels.append(mesh_full.point_data['node_label'] == 1)

        if i < len(graph_name_all) - 1:
            del mesh_full
    
    all_node_labels = np.array(all_node_labels)  # shape (4,N)
    all_node_mask = np.any(all_node_labels, axis=0)  # or function along the rows to yield N terms one for each node
    del all_node_labels

    points = mesh_full.points
    edges = mesh_full.lines.reshape(-1, 3)[:, 1:]

    mesh_lobe = pv.read(f'{data_dir}/{graph_name_lobe}.vtp')
    points_hull_lobe = mesh_lobe.points
    del mesh_lobe

    hull_idcs_full = np.where(all_node_mask == True)[0]
    hull_idcs_lobe = art.get_hull_idcs(points_hull_lobe, points)
    hull_idcs = np.unique(np.concatenate([hull_idcs_full, hull_idcs_lobe]))

    # do the refinement by converting any node connected to 2 surface nodes to a surface node
    hull_idcs_updated = np.copy(hull_idcs)

    diff = float('inf')
    count = 0

    print(hull_idcs_updated.shape)

    while diff != 0 and count < 10:

        count+=1
        new_hull_idcs,_ = art.connect_hull_points(np.arange(len(points)), hull_idcs_updated, edges)
        diff = len(new_hull_idcs)
        # print(len(new_hull_idcs))
        # print(new_hull_idcs)
        new_hull_idcs = np.array(new_hull_idcs)
        if diff != 0:
            hull_idcs_updated = np.concatenate((hull_idcs_updated, new_hull_idcs))

        print(f'count = {count}')
        print(f'number of new hull points = {diff}\n')

    # assign a label to each node indicating surface (1) or not (0)
    node_labels = np.zeros(len(points))
    node_labels[hull_idcs_updated] = 1
    mesh_full.point_data['node_label'] = node_labels

    # get the cortical depth for each node
    points_hull_updated = points[hull_idcs_updated]
    kd_tree = KDTree(points_hull_updated)
    distances, _ = kd_tree.query(points)

    mesh_full.point_data['cortical_depth_from_outer_wall'] = distances

    # also get the distance from the center for each node
    center = np.average(points_hull_updated,axis=0) # center is the mean of all the surface nodes
    mesh_full['distance_from_hull_center'] = np.linalg.norm(points-center, axis=1)
    
    # save the complete mesh
    mesh_full.save(f'{output_igraph_vtp_path}_fused.vtp')


if __name__ == '__main__':
    main()