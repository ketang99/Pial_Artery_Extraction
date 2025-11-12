'''
This file will take a vtp file with node labels and then convert that to a csv
The csv will 
'''

import sys

# directory that contains the py files
code_dir = 'C:\\Users\\Dell\\Documents\\RA_UBern\\code_for_chryso' 
sys.path.append(code_dir)

# home directory which has datasets, codes etc
home_dir = 'C:\\Users\\Dell\\Documents\\RA_UBern'

# full data directory that contains the vtp file to be scanned
data_dir = f'{home_dir}\\Oct_files'

# name of the vtp files without extension
# graph_name_full should be from hull_tweak_traversal_full.py and have 'all' in its name 
# this is because this has all the points and edges for the complete graph
graph_name_all = 'BL6_1_VesselVio_clean_fused_WITHdeg2_fused'

# desired output directory. Is set to data_dir currently
output_dir = data_dir
# desired output csv name. Script will save the result to data_dir
output_name = 'BL6_1_VesselVio_clean_fused_WITHdeg2' 
output_csv_path = f'{output_dir}/{output_name}_nodes'  # desired output vtp path excluding '.vtp'

sys.path.append(home_dir)
sys.path.append(code_dir)

import numpy as np
import pandas as pd
import pyvista as pv

def main():

    mesh = pv.read(f'{data_dir}/{graph_name_all}.vtp')
    points = mesh.points
    node_labels = mesh.point_data['node_labels']
    print(np.count_nonzero(node_labels==0), np.count_nonzero(node_labels==1))
    

    df_hull = pd.DataFrame()
    df_hull['NodeID'] = np.arange(len(points)).astype(int)
    df_hull['X'] = points[:,0]
    df_hull['Y'] = points[:,1]
    df_hull['Z'] = points[:,2]
    df_hull['node_labels'] = node_labels

    # save csv of the hull points to visualize via table to points on paraview
    df_hull.to_csv(f'{output_csv_path}.csv')
    print('hull csv saved\n')

if __name__ == '__main__':
    main()