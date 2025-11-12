import numpy as np
from copy import deepcopy
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def write_array(f, array, name, zeros=0, verbose=False):
    """Print arrays with different number of components, setting NaNs to 'substitute'.
    Optionally, a given number of zero-entries can be prepended to an
    array. This is required when the graph contains unconnected vertices.
    """
    tab = "  ";
    space = 5 * tab
    substituteD = -1000.;
    substituteI = -1000
    zeroD = 0.;
    zeroI = 0


    array_dimension = np.size(np.shape(array))

    if array_dimension > 1: # For arrays where attributes are vectors (e.g. coordinates)
        noc = np.shape(array)[1]
        firstel = array[0][0]
        Nai = len(array)
        Naj = np.ones(Nai,dtype=int)*noc
    else:
        noc = 1
        firstel = array[0]
        Nai = len(array)
        Naj = np.array([0], dtype='int')


    if type(firstel) == str:
        if verbose:
            print("WARNING: array '%s' contains data of type 'string'!" % name)
        return  # Cannot have string-representations in paraview.
    
    if "<type 'NoneType'>" in map(str, np.unique(np.array(map(type, array)))):
        if verbose:
            print("WARNING: array '%s' contains data of type 'None'!" % name)
        return
    
    
    if any([type(firstel) == x for x in
            [float, np.float32, np.float64, np.longdouble]]):
        atype = "Float64"
        format = "%f"
    elif any([type(firstel) == x for x in
            [int, np.int8, np.int16, np.int32, np.int64]]):
        atype = "Int64"
        format = "%i"
    else:
        if verbose:
            print("WARNING: array '%s' contains data of unknown type!" % name)
            print("k1")
        return

    f.write('{}<DataArray type="{}" Name="{}" '.format(4 * tab, atype, name))
    f.write('NumberOfComponents="{}" format="ascii">\n'.format(noc))

    if noc == 1:
        if atype == "Float64":
            for i in range(zeros):
                f.write('{}{}\n'.format(space, zeroD))
            aoD = np.array(array, dtype='double')
            for i in range(Nai):
                if not np.isfinite(aoD[i]):
                    f.write('{}{}\n'.format(space, substituteD))
                else:
                    f.write('{}{}\n'.format(space, aoD[i]))
        elif atype == "Int64":
            for i in range(zeros):
                f.write('{}{}\n'.format(space, zeroI))
            aoI = np.array(array, dtype=np.int64)
            for i in range(Nai):
                if not np.isfinite(aoI[i]):
                    f.write('{}{}\n'.format(space, substituteI))
                else:
                    f.write('{}{}\n'.format(space, aoI[i]))
    else:
        if atype == "Float64":
            atD = np.array(array, dtype='double')
            for i in range(zeros):
                f.write(space)
                for j in range(Naj[0]):
                    f.write('{} '.format(zeroD))
                f.write('\n')
            for i in range(Nai):
                f.write(space)
                for j in range(Naj[i]):
                    if not np.isfinite(atD[i, j]):
                        f.write('{} '.format(substituteD))
                    else:
                        f.write('{} '.format(atD[i, j]))
                f.write('\n')
        elif atype == "Int64":
            atI = np.array(array, dtype=np.int32)
            for i in range(zeros):
                f.write(space)
                for j in range(Naj[0]):
                    f.write('{} '.format(zeroI))
                f.write('\n')
            for i in range(Nai):
                f.write(space)
                for j in range(Naj[i]):
                    if not np.isfinite(atI[i, j]):
                        f.write('{} '.format(substituteI))
                    else:
                        f.write('{}'.format(atI[i, j]))
                f.write('\n')
    f.write('{}</DataArray>\n'.format(4 * tab))
#------------------------------------------------------------------------------------
def write_vtp(graph, filename, verbose=False, coordinatesKey='coords'):
    """Writes a graph in iGraph format to a vtp-file (e.g. for plotting with
    Paraview). Adds an index to both edges and vertices to make comparisons
    with the iGraph format easier.
    INPUT: graph: Graph in iGraph format
           filename: Name of the vtp-file to be written. Note that no filename-
                     ending is appended automatically.
           verbose: Whether or not to print to the screen if writing an array
                    fails.Default is False
           coordinatesKey: Key of vertex attribute in which the coordinates are stored. Default is 'coords'.
    OUTPUT: vtp-file written to disk.
    """

    # Make a copy of the graph so that modifications are possible, whithout
    # changing the original. Add indices that can be used for comparison with
    # the original, even after some edges / vertices in the copy have been
    # deleted:
    G = deepcopy(graph)
    G.vs['index'] = range(G.vcount())
    if G.ecount() > 0:
        G.es['index'] = range(G.ecount())

    # Delete selfloops as they cannot be viewed as straight cylinders and their
    # 'angle' property is 'nan':
    G.delete_edges(np.nonzero(G.is_loop())[0].tolist())

    tab = "  "
    fname = filename
    f = open(fname, 'w')

    # Find unconnected vertices:
    unconnected = np.nonzero([x == 0 for x in G.strength(weights=
                                                         [1 for i in range(G.ecount())])])[0].tolist()

    # Header
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="PolyData" version="0.1" ')
    f.write('byte_order="LittleEndian">\n')
    f.write('{}<PolyData>\n'.format(tab))
    f.write('{}<Piece NumberOfPoints="{}" '.format(2 * tab, G.vcount()))
    f.write('NumberOfVerts="{}" '.format(len(unconnected)))
    f.write('NumberOfLines="{}" '.format(G.ecount()))
    f.write('NumberOfStrips="0" NumberOfPolys="0">\n')

    # Vertex data
    keys = G.vs.attribute_names()
    keysToRemove = ['coords', 'pBC', 'rBC', 'kind', 'sBC', 'inflowE', 'outflowE', 'adjacent', 'mLocation', 'lDir',
                    'diameter']
    for key in keysToRemove:
        if key in keys:
            keys.remove(key)
    f.write('{}<PointData Scalars="Scalars_p">\n'.format(3 * tab))
    
    for key in keys:
        write_array(f, G.vs[key], key, verbose=verbose)
    f.write('{}</PointData>\n'.format(3 * tab))

    # Edge data
    keys = G.es.attribute_names()
    keysToRemove = ['diameters','lengths', 'lengths2', 'points', 'rRBC', 'tRBC', 'connectivity', "nkind"]
    for key in keysToRemove:
        if key in keys:
            keys.remove(key)
    f.write('{}<CellData Scalars="diameter">\n'.format(3 * tab))
    
    for key in keys:
        write_array(f, G.es[key], key, zeros=len(unconnected), verbose=verbose)
    f.write('{}</CellData>\n'.format(3 * tab))

    # Vertices
    f.write('{}<Points>\n'.format(3 * tab))
    write_array(f, np.vstack(G.vs[coordinatesKey]), coordinatesKey, verbose=verbose)
    f.write('{}</Points>\n'.format(3 * tab))

    # Unconnected vertices
    if unconnected != []:
        f.write('{}<Verts>\n'.format(3 * tab))
        f.write('{}<DataArray type="Int64" '.format(4 * tab))
        f.write('Name="connectivity" format="ascii">\n')
        for vertex in unconnected:
            f.write('{}{}\n'.format(5 * tab, vertex))
        f.write('{}</DataArray>\n'.format(4 * tab))
        f.write('{}<DataArray type="Int64" '.format(4 * tab))
        f.write('Name="offsets" format="ascii">\n')
        for i in range(len(unconnected)):
            f.write('{}{}\n'.format(5 * tab, 1 + i))
        f.write('{}</DataArray>\n'.format(4 * tab))
        f.write('{}</Verts>\n'.format(3 * tab))

    # Edges
    f.write('{}<Lines>\n'.format(3 * tab))
    f.write('{}<DataArray type="Int64" '.format(4 * tab))
    f.write('Name="connectivity" format="ascii">\n')
    for edge in G.get_edgelist():
        f.write('{}{} {}\n'.format(5 * tab, edge[0], edge[1]))
    f.write('{}</DataArray>\n'.format(4 * tab))
    f.write('{}<DataArray type="Int64" '.format(4 * tab))
    f.write('Name="offsets" format="ascii">\n')
    for i in range(G.ecount()):
        f.write('{}{}\n'.format(5 * tab, 2 + i * 2))
    f.write('{}</DataArray>\n'.format(4 * tab))
    f.write('{}</Lines>\n'.format(3 * tab))

    # Footer
    f.write('{}</Piece>\n'.format(2 * tab))
    f.write('{}</PolyData>\n'.format(1 * tab))
    f.write('</VTKFile>\n')
    

    f.close()

