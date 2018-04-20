"""tree utility"""

import numpy as np
from numpy import linalg as LA
import math
from scipy.sparse import csr_matrix
from copy import deepcopy
from numpy.linalg import inv
from sklearn import preprocessing

def branch_order(parent_index):
    """
    Set the branching numbers. It gives bach a vector with the length of
    number of nodes such that in each index of this vector the branching
    number of this node is written: terminal = 0, passig = 1, branching = 2
    dependency:
        nodes_list
    """
    branch_order = np.zeros(len(parent_index))
    unique, counts = np.unique(parent_index[1:], return_counts=True)
    branch_order[unique] = counts
    return branch_order

def dendogram_depth(parent_index):
    depth = np.zeros(len(parent_index))
    up = np.arange(len(parent_index))
    while(sum(up) != 0):
        up = parent_index[up]
        depth = np.sign(up) + depth
    depth += 1
    depth[0] = 0
    return depth

def get_parent_index(input_parent, input_type='swc_matrix'):
    if input_type=='swc_matrix':
        parent_index = input_parent[:,6]-1
        parent_index[0] = 0
        parent_index = parent_index.astype(int)

    if input_type=='Neuron':
        parent_index = input_parent.parent_index

    return parent_index

def get_index_of_critical_points(input_parent,
                            input_type='swc_matrix',
                            n_soma=3,
                            only_one_somatic_node=True):
    """
    Returning the index of branching points and end points.

    Parameters
    ----------
    input_parent: the data of parent
        it can have different type according to input_type
    input_type: str
        it can be 'swc_matrix, 'Neuron_no_featured', 'Neuron_featured' 'parent_index', 'branch_order'
    n_soma: int
        the number of nodes that represent soma
    only_one_somatic_node: boolean
        True means that only  one node will represent soma

    Returns
    -------
    critical_nodes: numpy array
        the list of critical points; branching points and end points
    """
    if input_type=='branch_order':
        branch_index = input_parent

    if input_type=='swc_matrix':
        parent_index = input_parent[:,6]-1
        parent_index[0] = 0
        parent_index = parent_index.astype(int)
        n_soma = len(np.where(input_parent[:,1] == 1)[0])
        branch_order = np.zeros(len(parent_index))
        unique, counts = np.unique(parent_index[1:], return_counts=True)
        branch_order[unique] = counts

    if input_type=='Neuron_featured':
        branch_order = input_parent.features['branch order']
        n_soma = input_parent.n_soma

    if input_type=='Neuron_no_featured':
        parent_index = input_parent.parent_index
        n_soma = input_parent.n_soma
        branch_order = np.zeros(len(parent_index))
        unique, counts = np.unique(parent_index[1:], return_counts=True)
        branch_order[unique] = counts

    if input_type=='parent_index':
        parent_index = input_parent
        branch_order = np.zeros(len(parent_index))
        unique, counts = np.unique(parent_index[1:], return_counts=True)
        branch_order[unique] = counts

    if only_one_somatic_node is False:
        n_soma = 1

    (branch_index,) = np.where(branch_order[n_soma:] == 2)
    (endpoint_index,) = np.where(branch_order[n_soma:] == 0)
    selected_index = np.union1d(branch_index + n_soma,
                                endpoint_index + n_soma)
    main_index = np.append(0, selected_index)
    return main_index


def parent_id(whole_tree, node_on_subtree, some_selected_node_on_subtree):
    """
    Return the parent id of all the selected_index of the neurons.

    Parameters
    ----------
    selected_index: numpy array
        the index of nodes

    Returns
    -------
    parent_id: the index of parent of each element in selected_index in
    this array.
    """
    parent_id = np.array([], dtype=int)
    for i in selected_index:
        p = parent_index[i]
        while(~np.any(selected_index == p)):
            p = parent_index[p]
        (ind,) = np.where(selected_index == p)
        parent_id = np.append(parent_id, ind)
    return parent_id
