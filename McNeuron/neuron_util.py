"""Collection of Utility Functions for Neuron"""
import numpy as np
import scipy
from McNeuron import Neuron
from McNeuron import Node
from sklearn import preprocessing


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
                spherical=False,
                length=10):
    """
    Make a star-wise neuron. The location of the root is origin.

    Parameters:
    wing_number: int
        The number of blades that the neuron should have
    node_on_each_wings: int
        The number of nodes on each warnings

    Return:
    -------
    neuron: Neuron
        The desire neuron
    """
    nodes_list = []
    root = Node()
    root.r = 1.
    root.node_type = 1
    root.xyz = np.array([0, 0, 0], dtype=float)
    nodes_list.append(root)

    for i in range(0):
        soma = Node()
        soma.r = .2
        soma.node_type = 1
        soma.xyz = np.array([0, 0, 0], dtype=float)
        nodes_list.append(soma)
        root.add_child(soma)
        soma.parent = root

    angle = 2 * np.pi/wing_number
    for j in range(wing_number):
        rand_vec = np.random.randn(3)
        rand_vec = rand_vec/np.sqrt(sum(rand_vec**2))
        for i in range(node_on_each_wings):
            node = Node()
            node.r = .2
            node.node_type = 2
            if spherical:
                x = rand_vec[0] * length * (i+1)
                y = rand_vec[1] * length * (i+1)
                z = rand_vec[2] * length * (i+1)
            else:
                x = np.sin(j*angle) * length * (i+1)
                y = np.cos(j*angle) * length * (i+1)
                z = 0.
            node.xyz = np.array([x, y, z], dtype=float)  # +0*np.random.rand(3)
            if i == 0:
                root.add_child(node)
                node.parent = root
                nodes_list.append(node)
            else:
                nodes_list[-1:][0].add_child(node)
                node.parent = nodes_list[-1:][0]
                nodes_list.append(node)
    neuron = Neuron(input_format='only list of nodes', input_file=nodes_list)
    return neuron


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
               input_format='Matrix of swc')
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
