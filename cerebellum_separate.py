import sys

# directory that contains the py files
code_dir = '' 
sys.path.append(code_dir)

# home directory which has datasets, codes etc
home_dir = '' 

# full data directory that contains the vtp file
data_dir = ''
# name of the vtp file
graph_name = ''

# the csv file which contains the details about the cerebellum boundaries
# the directory where it is stored
csv_dir = ''
# the name of the file without .csv
csv_name = ''
# margin for the cerebellum separation
margin = 20

'''
This margin assumes the orientation of the brain is such that the cerebellum is vertically above the rest of the brain
This is to say that the cerebellum is along the positive Y direction

'''

# desired output directory. Is set to data_dir currently
output_dir = data_dir

# desired output name excluding .vtp. Script will save the result to data_dir
output_name = '' 
output_igraph_vtp_path = f'{output_dir}/{output_name}'  # desired output vtp path excluding '.vtp'

sys.path.append(home_dir)
sys.path.append(code_dir)

import numpy as np
import pandas as pd
import igraph as ig
import pyvista as pv

from write_vtp_py3 import write_vtp

from utils import *


def identify_line(point, xlims):
    x = point[0]
    if x < xlims[0,0]:
        ind = 0
    elif x > xlims[-1,-1]:
        ind = len(xlims)-1
    else:
        boolarr = np.logical_and(x>=xlims[:,0], x<=xlims[:,1])
        ind = np.where(boolarr!=0)[0][0]

    return ind

def separate_cerebellum(points, lines, xlims, margin=0.0):

    id_cerebellum = []
    id_remainder = []

    for i,p in enumerate(points):
        ind = identify_line(p, xlims)
        line = lines[ind]
        y_calc = p[0]*line[0] + line[1]
        if p[1] <= y_calc + margin:
            id_remainder.append(i)
        else:
            id_cerebellum.append(i)

    return id_cerebellum, id_remainder


def main():

    # read the main vtp file
    mesh = pv.read(f'{data_dir}/{graph_name}.vtp')
    points = mesh.points
    edges = mesh.lines.reshape(-1, 3)[:, 1:]

    # Read the cerebellum CSV
    df = pd.read_csv(f'{csv_dir}/{csv_name}.csv', header=None)
    df.columns = ['p0_coord0', 'p0_coord1', 'p1_coord0', 'p1_coord1']

    df = df.astype(float)
    # Compute slope (m) and intercept (c) for each row
    m = (df['p1_coord1'] - df['p0_coord1']) / (df['p1_coord0'] - df['p0_coord0'])
    c = df['p0_coord1'] - m * df['p0_coord0']

    # Stack into an [n, 2] array
    lines = np.column_stack((m, c))

    xlims = [df['p0_coord0'], df['p1_coord0']]
    xlims = np.array(xlims)
    xlims = xlims.T

    id_c, _ = separate_cerebellum(points, lines, xlims, margin)

    labels = np.zeros(len(points)).astype(np.uint8)
    labels[id_c] = 1
    mesh.point_data['cerebellum_label'] = labels
    mesh.save(f'{output_igraph_vtp_path}.vtp')
    print('vtp with cerebellum label saved')


if __name__ == '__main__':
    main()