"""Collection of Utility Functions for Neuron"""
import numpy as np
import scipy
from sklearn import preprocessing

def add_soma_if_needed(swc_matrix):
    if len(swc_matrix) == 0:
        swc_matrix = np.array([[1,1,0,0,0,1,-1],[2,1,0,0,1,1,1]])
    elif (swc_matrix[0,0] != 1):
        swc_matrix = np.append(np.array([[1,1,0,0,0,1,-1]]), swc_matrix, axis=0)
    return swc_matrix

def repair_triple(swc_matrix):
    """
    Remove the triple node (nodes with the branching more than 2)
    and reconnect them to the parent of tirple node (or possibily grand parent ,...)
    """
    original_parent = swc_matrix[:, 6].astype(int)-1
    original_parent[0] = 0
    branch_order = np.zeros(swc_matrix.shape[0])
    unique, counts = np.unique(original_parent[1:], return_counts=True)
    branch_order[unique.astype(int)] = counts
    (triple_nodes,) = np.where(branch_order[1:]>2)
    new_swc_matrix = deepcopy(swc_matrix)
    if len(triple_nodes) is not 0:
        node = triple_nodes[0] + 1
        (child_triple,) = np.where(original_parent == node)
        new_swc_matrix[child_triple[0], 6] = original_parent[node] + 1
        new_swc_matrix = repair_triple(new_swc_matrix)
    return new_swc_matrix

def correct_swc(swc_matrix):
    """
    Correct swc if it is fixable. The issues that can be fixed are:
        - adding a node with index 1 as the root if it does not exist
        - if the tree structure is not binary makes it binary.
    """
    swc_matrix = add_soma_if_needed(swc_matrix)
    swc_matrix = repair_triple(swc_matrix)
    return swc_matrix

def get_standard_order(parents):
    """
    Reorder a given parents sequence.
    Parent labels < children labels.

    Parameters
    ----------
    parents: numpy array
        sequence of parents indices
        starts with -1
    locations: numpy array
        n - 1 x 3

    Returns
    -------
    parents_reordered: numpy array
        sequence of parents indices
    locations_reordered: numpy array
        n - 1 x 3
    """
    parents = np.array(parents, dtype=int)
    parents = np.squeeze(parents)
    length = len(parents)
    # Construct the adjacency matrix
    adjacency = np.zeros([length, length])
    adjacency[parents[1:]-1, range(1, length)] = 1

    # Discover the permutation with Schur decomposition
    full_adjacency = np.linalg.inv(np.eye(length) - adjacency)
    full_adjacency_permuted, permutation_matrix = \
        scipy.linalg.schur(full_adjacency)

    # Reorder the parents
    parents_reordered = \
        np.argmax(np.eye(length) - np.linalg.inv(full_adjacency_permuted),
                  axis=0) + 1
    parents_reordered[0] = -1
    return parents_reordered, permutation_matrix.T

def correct_swc(swc_matrix):

    if swc_matrix.shape[0] <2:
        return np.array([[1,1,0,0,0,1,-1],[2,1,0,0,1,1,1]])
    else:
        new_swc = swc_matrix
        new_swc[0,1] = 1
        new_swc[new_swc[:,6]==-1,6] = 1
    return new_swc

def make_standard_swc(matrix):
    parents_reordered, permutation_matrix = get_standard_order(matrix[:, 6])
    new_matrix = np.zeros_like(matrix)
    new_matrix[:, 6] = parents_reordered
    new_matrix[:, 1:6] = np.dot(permutation_matrix, matrix[:, 1:6])
    return new_matrix

def check_ordering(parents):
    """
    Checking the ordering of the parents. Return zero is it's standard, and
    non-zero otherwise.
    """
    parents = np.squeeze(parents)
    (I, ) = np.where(parents - range(0, len(parents)) > 0)
    return sum(I)


def star_neuron(wing_number=3,
                node_on_each_wings=4,
                length=10,
                orientation = np.array([1,0,0])):
    """
    Make a star-wise neuron. The location of the root is origin.

    Parameters:
    wing_number: int
        The number of blades that the neuron should have
    node_on_each_wings: int
        The number of nodes on each warnings

    Return:
    -------
    neuron: swc matrix of a star shape neuron
    """
    n_node = wing_number*node_on_each_wings+1
    swc_matrix = np.zeros([n_node, 7])
    swc_matrix[:,0] = np.arange(n_node)+1.
    swc_matrix[0,6] = -1.
    swc_matrix[0,1] = 1.
    swc_matrix[1:,1] = 3.
    swc_matrix[:,5]  = 1.
    index = 1
    angle = 2 * np.pi/wing_number
    for wg in range(wing_number):
        for n in range(node_on_each_wings):
            swc_matrix[index, 2] =  length * (n+1) * (orientation[0] * np.cos(wg*angle) - orientation[1] * np.sin(wg*angle))
            swc_matrix[index, 3] =  length * (n+1) * (orientation[0] * np.sin(wg*angle) + orientation[1] * np.cos(wg*angle))
            swc_matrix[index, 4] = 0
            if n == 0:
                swc_matrix[index, 6] = 1
            else:
                swc_matrix[index, 6] = index
            index += 1
    return swc_matrix

def make_standard(neuron):
    """
    Make the neuron standard. i.e. the index of parent < children
    """
    p_reordered, per = get_standard_order(neuron.parent_index+1)
    order = np.argmax(per, axis=0)
    neuron.parent_index = p_reordered - 1
    neuron.branch_order = np.dot(per, neuron.branch_order)
    neuron.nodes_list = [neuron.nodes_list[i] for i in order]
    neuron.global_angle = np.dot(per, neuron.global_angle)
    neuron.distance_from_parent = np.dot(per, neuron.distance_from_parent)
    neuron.distance_from_root = np.dot(per, neuron.distance_from_root)
    neuron.connection = np.dot(per, neuron.connection)
    neuron.branch_angle = np.dot(per, neuron.branch_angle.T).T
    neuron.local_angle = np.dot(per, neuron.local_angle.T).T
    neuron.location = np.dot(per, neuron.location.T).T

    return neuron


def check_neuron(neuron):
    """
    Check the features of the neurons.

    Parameters:
    -----------
    neuron: Neuron
        The neuron to be checked.

    Returns:
    warnings if some features are not correct.
    """
    cor = 1
    n = Neuron(input_file=get_swc_matrix(neuron),
               input_format='Matrix of swc without Node class')
    n.fit()
    list_features = neuron.features.keys()
    for f in range(len(list_features)):
        # if list_features[f] == 'curvature':
        #     print(n.features[list_features[f]])
        #     print(neuron.features[list_features[f]])
        #     print(n.features[list_features[f]] - neuron.features[list_features[f]])
        if len(n.features[list_features[f]]) -  \
                len(neuron.features[list_features[f]]) != 0:
            print("The size of feature " + list_features[f] +
                  " is not calculated correctly.")
            cor = 0
        else:
            a = n.features[list_features[f]]-neuron.features[list_features[f]]
            if list_features[f] == 'branch_angle' or list_features[f] == 'side_branch_angle':
                a = np.sort(n.features[list_features[f]], axis=0) - \
                    np.sort(neuron.features[list_features[f]], axis=0)
            a = a**2
            # if list_features[f] == 'curvature':
            #     print(n.features[list_features[f]])
            #     print(neuron.features[list_features[f]])
            #     print(a.sum())
            if(a.sum() > 0.):
                print("The feature "+list_features[f] +
                      " is not calculated correctly.")
                cor = 0
    # if cor == 1:
        # print("Neuron's attributes seem to be correct.")


def get_swc_matrix(neuron):
    """
    Gets back the swc format for the neuron.
    """
    loc = neuron.location
    A = np.zeros([loc.shape[1], 7])
    A[:, 0] = np.arange(loc.shape[1])
    A[:, 1] = neuron.nodes_type
    A[:, 2:5] = loc.T
    A[:, 5] = neuron.diameter
    A[:, 6] = neuron.parent_index + 1
    A[0, 6] = -1
    return A


def write_swc(self, neuron):
    """
    Write neuron in swc file. Used to write an SWC file from a morphology
    stored in this :class:`Neuron`.

    """
    writer = open("neuron", 'w')
    swc = get_swc_matrix(neuron)
    for i in range(swc.shape[0]):
        string = (str(swc[i, 0])+' '+str(swc[i, 1]) + ' ' + str(swc[i, 2]) +
                      ' ' + str(swc[i, 3]) + ' ' + str(swc[i, 4]) +
                      ' ' + str(swc[i, 5]) + ' ' + str(swc[i, 6]))
        writer.write(string + '\n')
        writer.flush()
    writer.close()
    return writer


def important_node_full_matrix(neuron):
    lines = []
    (branch_index,)  = np.where(neuron.branch_order==2)
    (end_nodes,)  = np.where(neuron.branch_order==0)
    important_node = np.append(branch_index,end_nodes)
    parent_important = neuron.parent_index_for_node_subset(important_node)
    important_node = np.append(0, important_node)
    L = []
    for i in parent_important:
        (j,) = np.where(important_node==i)
        L = np.append(L,j)
    matrix = np.zeros([len(L),len(L)])
    for i in range(len(L)):
        if(L[i]!=0):
            matrix[i,L[i]-1] = 1
    B = inv(np.eye(len(L)) - matrix)
    return B

def decompose_immediate_children(matrix):
    """
    Parameters
    ----------
    matrix : numpy array of shape (n,n)
        The matrix of connetion. matrix(i,j) is one is j is a grandparent of i.

    Return
    ------
    L : list of numpy array of square shape
        L consists of decomposition of matrix to immediate children of root.
    """
    a = matrix.sum(axis = 1)
    (children,) = np.where(a == 1)
    L = []
    #print(children)
    for ch in children:
        (ind,) = np.where(matrix[:, ch] == 1)
        ind = ind[ind != ch]
        L.append(matrix[np.ix_(ind, ind)])
    p = np.zeros(len(L))
    for i in range(len(L)):
        p[i] = L[i].shape[0]
    s = np.argsort(p)
    List = []
    for i in range(len(L)):
        List.append(L[s[i]])
    return List

def read_hoc_format(file_address):
    """
    Parameters
    ----------
    file_address: str
        the address of of hoc file (text file)
    Returns
    -------
    swc_matrix: numpy array
        the matrix of swc
    """
    i = 0
    a = open(file_address)
    initials = []
    ends = []
    for line in a:
        if line.startswith('connect'):
            split = line.split()
            initials.append(split[1][:-4])
            ends.append(split[2][:-3])

    ends = list(ends)
    initials = list(initials)
    alls = list(set(ends + initials))
    len_all = len(alls)
    A = np.zeros([len_all, len_all])

    le = preprocessing.LabelEncoder()
    le.fit(alls)
    A[le.transform(initials), le.transform(ends)] = 1

    par, per = \
        scipy.linalg.schur(A.T)
    par = par.T
    per = per.T
    par_ind = list(np.argmax(per, axis=1))
    child_ind = list(np.argmax(per, axis=0))

    swc_matrix = np.zeros([len_all, 7])
    a = open(file_address)
    i = 0
    bol = 1
    x = 0.
    y = 0.
    z = 0.
    r = 0.
    for line in a:
        if(bol < 0):
            bol += 1
        elif bol == 0:
            split = line.split()
            if(split[0] != '}'):
                x = np.append(x, float(split[0][8:-1]))
                y = np.append(y, float(split[2][:-1]))
                z = np.append(z, float(split[1][:-1]))
                r = np.append(r, float(split[3][:-1]))
            else:
                ind = le.transform([name])[0]
                ind2 = child_ind[ind]
                child_index = ind2
                ind3 = np.where(par[ind2, :] == 1)[0]
                if(len(ind3) == 0):
                    parent_index = -1
                else:
                    parent_index = ind3+1

                swc_matrix[child_index, 0] = child_index
                swc_matrix[child_index, 1] = t
                swc_matrix[child_index, 2] = x[1:].mean()
                swc_matrix[child_index, 3] = y[1:].mean()
                swc_matrix[child_index, 4] = z[1:].mean()
                swc_matrix[child_index, 5] = r[1:].mean()
                swc_matrix[child_index, 6] = parent_index
                bol += 1
                x = 0.
                y = 0.
                z = 0.
                r = 0.
        else:
            split = line.split()
            try:
                ind = alls.index(split[0])
                if(split[1] == '{'):
                    name = split[0]
                    bol = -2
                    if('soma' in split[0]):
                        t = 1
                    elif('dendrite' in split[0]):
                        t = 3
                    elif('axon' in split[0]):
                        t = 2
                    else:
                        print('undefined type for:')
                        print name
            except:
                i = 0
    swc_matrix = make_standard_swc(swc_matrix)
    return swc_matrix

def permute_indexing(neuron, permutation):
    swc = np.zeros([neuron.n_node, 7])
    swc[:, 2:5] = neuron.location[:, permutation].T
    swc[:, 1] = neuron.nodes_type[permutation]
    swc[:, 5] = neuron.diameter[permutation]
    inv = np.argsort(permutation)
    swc[:, 6] = inv[neuron.parent_index[permutation]]+1
    swc[0, 6] = -1
    return swc

def length_first_indexing(neuron):
    permutation = np.argsort(neuron.neural_distance_from_root(neuron.distance_from_parent()))
    swc = permute_indexing(neuron, permutation)
    return swc

def dendogram_depth(parent_index):
    ancestor_matrix = np.arange(len(parent_index))
    ancestor_matrix = np.expand_dims(ancestor_matrix, axis=0)
    up = np.arange(len(parent_index))
    while(sum(up) != 0):
        up = parent_index[up]
        ancestor_matrix = np.append(ancestor_matrix,
                                    np.expand_dims(up, axis=0),
                                    axis=0)
    return np.sign(ancestor_matrix[:-1, :]).sum(axis=0)

def map_to_zero_one(vector):
    """
    Mapping linearly the values in a 2 dimesion vector to [0,1] interval.

    Parameters:
    -----------
    vector: 2d numpy

    Return:
    -------
    scaled_vector: numpy
        an array with the same size of vector, that its values are mapped between [0, 1]
        linearly.
    """
    lowest, highest = vector.min(axis=1), vector.max(axis=1)
    length = highest - lowest
    length[length==0] = 1
    scaled_vector = (vector - np.array([lowest]).T)/np.array([length]).T
    return scaled_vector
