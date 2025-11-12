'''
This script will allow the user to read csv files for edges and nodes,
and then save a graph as a vtp file

The saved vtp file will have:
- the node coordinates, IDs and connectivity
- edge diameters, lengths and other edge attributes etc.

The saved vtp file will also contain labels indicating which graph component each node belongs to 
'''

import sys
from collections import Counter

# directory that contains the py files
code_dir = '' 
sys.path.append(code_dir)

# home directory which has datasets, codes etc
home_dir = '' 
sys.path.append(home_dir)

# directory of the dataset folder which contains the edges/nodes csv files
data_dir = ''

# filenames of csv's containing edges and nodes excluding '.csv'.
csv_name_edges = ''
csv_name_nodes = ''

# desired output directory. Is set to data_dir currently
output_dir = data_dir

# desired output name
output_name = '' 
output_igraph_vtp_path = f'{output_dir}/{output_name}'  # desired output vtp path excluding '.vtp'

import numpy as np
import pandas as pd
import igraph
import pyvista as pv
from write_vtp_py3 import write_vtp
from scipy.spatial import KDTree


def main():

    df_node = pd.read_csv(f'{data_dir}/{csv_name_nodes}.csv', delimiter=';')
    coords = np.abs(df_node[['pos_x', 'pos_y', 'pos_z']].values)

    df_edges = pd.read_csv(f'{data_dir}/{csv_name_edges}.csv', delimiter=';')
    edge_list = df_edges[['node1id', 'node2id']].values

    graph = igraph.Graph(edge_list.tolist())  # igraph object

    # write the edge attributes to the graph
    for col in df_edges.columns:
        if col not in ["node1id", "node2id"]:  # skip source/target columns
            graph.es[col] = df_edges[col].values.tolist()

    # node IDs that user wants to delete
    nodes_to_delete = []

    # delete the above node IDs
    if len(nodes_to_delete) != 0:
        neighbors_node = []
        incident_edges = []
        for node in nodes_to_delete:
            if not node in neighbors_node:
                neighbors_node.append(node)

            id_nodes = graph.neighbors(node)
            for id_n in id_nodes:
                if not id_n in neighbors_node:
                    neighbors_node.append(id_n)

            id_edges = graph.incident(node)
            for id_e in id_edges:
                if not id_e in incident_edges:
                    incident_edges.append(id_e)

        graph.delete_vertices(neighbors_node)
        graph.delete_edges(incident_edges)

        new_coords = np.delete(coords, neighbors_node, axis=0)
        graph.vs["coords"] = new_coords

    else:
        graph.vs["coords"] = coords

    # find the minimum possible distance between two nodes in thh graph
    tree = KDTree(coords)

    # Query the two nearest neighbors for each point (itself + the closest other point)
    distances, indices = tree.query(coords, k=2)

    # distances[:, 0] is always 0 (distance to itself), so take the second column
    nearest_neighbor_dist = distances[:, 1]

    # Get the minimum distance among all those nearest-neighbor distances
    min_distance = nearest_neighbor_dist.min()

    # TAKE NOTE OF THIS VALUE
    print("Minimum nearest-neighbor distance:", min_distance)

    # obtain the unconnected components of the graph
    components = graph.connected_components()

    # arrange the components in descending order of len(components[i])
    component_sizes = []
    for c in components:
        component_sizes.append(len(c))
    component_sizes = np.array(component_sizes)

    sorted_sizes = np.argsort(component_sizes)[-1::-1]
    component_labels = np.zeros(len(graph.vs["coords"]))
    for i,s in enumerate(sorted_sizes):
        component_labels[components[i]] = i

    # counter = Counter(component_labels)
    # top_k_values = [val for val, _ in counter.most_common(len(components))]

    # component_labels = np.zeros((len(graph.vs["coords"]),), dtype=int)
    # for i,c in enumerate(top_k_values):
        # component_labels[components[c]] = i

    graph.vs["component_label"] = component_labels

    # save the graph as a vtp file
    write_vtp(graph, f'{output_igraph_vtp_path}_all_components.vtp', coordinatesKey='coords')

    # save just the 32 largest components if at least that many components exist

    if len(components) > 32:

        k = 31

        print('More than 32 components exist so a vtp file highlighting the largest 32 is being created')

        # Step 1: Find the top-k most frequent values
        counter = Counter(component_labels)
        top_k_values = [val for val, _ in counter.most_common(k)]

        # Step 2: Initialize the output array
        largest_component_labels = k * np.ones_like(component_labels)

        # Step 3: For each top-k value, assign its rank index to the corresponding positions
        for i, val in enumerate(top_k_values):
            indices = np.where(component_labels == val)[0]
            largest_component_labels[indices] = i

        graph.vs["component_label"] = largest_component_labels

        write_vtp(graph, f'{output_igraph_vtp_path}_top{k+1}_components.vtp', coordinatesKey='coords')


if __name__ == '__main__':
    main()