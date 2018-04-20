"""Collection of subsampling method on the neurons."""
import numpy as np
from McNeuron import Neuron
from numpy import linalg as LA
from copy import deepcopy
from McNeuron import swc_util
from McNeuron import tree_util

def parent_id(parent_index, selected_index):
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

def swc_matrix_for_selected_nodes(swc_matrix, selected_index):
    """
    Giving back a new neuron made up with the selected_index nodes of self.
    if node A is parent (or grand parent) of node B in the original neuron,
    it is the same for the new neuron.

    Parameters
    ----------
    selected_index: numpy array
        the index of nodes from original neuron for making new neuron.

    Returns
    -------
    Neuron: the subsampled neuron.
    """
    original_parent_index = swc_matrix[:,6] - 1
    original_parent_index[0] = 0
    parent = parent_id(original_parent_index, selected_index)
    selected_index_swc_matrix = swc_matrix[selected_index, :]
    selected_index_swc_matrix[:, 6] = parent + 1
    selected_index_swc_matrix[0, 6] = -1 
    return selected_index_swc_matrix

def straigh_subsample(swc_matrix, length):
    """
    Subsampling a neuron from original neuron. It has all the main points of the original neuron,
    i.e endpoints or branching nodes, and meanwhile the distance of two consecutive nodes
    of subsample neuron is around the 'distance'.
    for each segment between two consecuative main points, a few nodes from the segment will be added to the selected node;
    it starts from the far main point, and goes on the segment toward the near main point. Then the first node which is
    going to add has the property that it is the farest node from begining on the segment such that its distance from begining is
    less than 'distance'. The next nodes will be selected similarly. this procesure repeat for all the segments.

    Parameters
    ----------
    distance: float
        the mean distance between pairs of consecuative nodes.

    Returns
    -------
    Neuron: the subsampled neuron
    """
    swc_matrix = swc_util.correct_swc(swc_matrix)
    parent_index = swc_matrix[:,6]-1
    parent_index[0] = 0
    parent_index = parent_index.astype(int)
    n_soma = len(np.where(swc_matrix[:,1] == 1)[0])
    selected_index = tree_util.get_index_of_main_nodes(swc_matrix)
    
    a = (swc_matrix[:, 2:5] - swc_matrix[parent_index, 2:5]) ** 2
    distance_from_parent = np.sqrt(a.sum(axis=1))
    # for each segment between two consecuative main points, a few nodes from the segment will be added to the selected node.
    # These new nodes will be selected base on the fact that neural distance of two consecuative nodes is around 'distance'.
    # Specifically, it starts from the far main point, and goes on the segment toward the near main point. Then the first node which is
    # going to add has the property that it is the farest node from begining on the segment such that its distance from begining is
    # less than 'distance'. The next nodes will be selected similarly.
    for i in selected_index:
        upList = np.array([i], dtype=int)
        index = parent_index[i]
        dist = distance_from_parent[i]
        while(~np.any(selected_index == index)):
            upList = np.append(upList, index)
            index = parent_index[index]
            dist = np.append(dist, sum(distance_from_parent[upList]))
        dist = np.append(0, dist)
        (I,) = np.where(np.diff(np.floor(dist/length)) > 0)
        I = upList[I]
        selected_index = np.append(selected_index, I)
    selected_index = np.unique(selected_index)
    neuron = swc_matrix_for_selected_nodes(swc_matrix, selected_index)
    return neuron

def get_index_of_main_nodes(swc_matrix):
    """
    Returning the index of branching points and end points.

    Parameters
    ----------
    neuron: Neuron
        input neuron

    Returns
    -------
    selected_index: array
        the list of main point; branching points and end points
    """
    parent_index = swc_matrix[:,6]-1
    parent_index[0] = 0
    parent_index = parent_index.astype(int)
    n_soma = len(np.where(swc_matrix[:,1] == 1)[0])
    branch_order = np.zeros(len(parent_index))
    unique, counts = np.unique(parent_index[1:], return_counts=True)
    branch_order[unique] = counts
    (branch_index,) = np.where(branch_order[n_soma:] == 2)
    (endpoint_index,) = np.where(branch_order[n_soma:] == 0)
    selected_index = np.union1d(branch_index + n_soma,
                                endpoint_index + n_soma)
    selected_index = np.append(0, selected_index)
    return selected_index

def regular_subsample(swc_matrix):
    """
    Returning subsampled neuron with main nodes. i.e endpoints and branching
    nodes.

    Parameters
    ----------
    neuron: Neuron
        input neuron

    Returns
    -------
    Neuron: the subsampled neuron
    """
    swc_matrix = swc_util.correct_swc(swc_matrix)
    selected_index = get_index_of_main_nodes(swc_matrix)
    selected_index = np.unique(selected_index)
    neuron = swc_matrix_for_selected_nodes(swc_matrix, selected_index)
    return neuron

def fast_straigthen_subsample_swc(swc_matrix, length):
    swc_matrix = swc_util.correct_swc(swc_matrix)
    swc_matrix = repair_triple(swc_matrix)
    original_parent = swc_matrix[:, 6].astype(int)-1
    original_parent[0] = 0
    n_soma = len(np.where(swc_matrix[:,1] == 1)[0])
    try:
        index_of_main_nodes = _get_index_of_main_nodes(n_soma, original_parent).astype(int)
        n_selected_node_on_each_segment, selected_index = \
            _get_index_node_straighten(index_of_main_nodes,
                                      original_parent,
                                      swc_matrix,
                                      length)

        n_all_nodes = len(index_of_main_nodes)+int(sum(n_selected_node_on_each_segment))

        par_main = _get_parent_main(original_parent, index_of_main_nodes).astype(int)

        index_main_in_all = np.cumsum(n_selected_node_on_each_segment+1).astype(int)
        index_main_in_all = np.append(0, index_main_in_all)+1
        index_main_in_all = np.append(0, index_main_in_all)

        par = np.arange(n_all_nodes)
        par_index_main_in_all = index_main_in_all[par_main+1]
        par[index_main_in_all[:-1]] = par_index_main_in_all
        par[0] = -1

        new_swc_matrix = np.zeros([len(par),7])
        selected_index = np.sort(np.append(selected_index, index_of_main_nodes))
        new_swc_matrix[:, 6] = par
        new_swc_matrix[:, 1:6] = swc_matrix[selected_index, 1:6]
    except:
        new_swc_matrix = straigh_subsample(swc_matrix, length)            
    return new_swc_matrix


def find_sharpest_fork(nodes):
    """
    Looks at the all branching point in the Nodes list, selects those which
    both its children are end points and finds the closest pair of childern
    (the distance between children).

    Parameters
    ----------
    Nodes: list
    the list of Node

    Returns
    -------
    sharpest_pair: array
        the index of the pair of closest pair of childern
    distance: float
        Distance of the pair of children
    """
    pair_list = []
    Dis = np.array([])
    for n in nodes:
        if n.parent is not None:
            if n.parent.parent is not None:
                a = n.parent.children
                if(isinstance(a, list)):
                    if(len(a)==2):
                        n1 = a[0]
                        n2 = a[1]
                        if(len(n1.children) == 0 and len(n2.children) == 0):
                            pair_list.append([n1 , n2])
                            dis = LA.norm(a[0].xyz - a[1].xyz,2)
                            Dis = np.append(Dis,dis)
    if(len(Dis)!= 0):
        (b,) = np.where(Dis == Dis.min())
        sharpest_pair = pair_list[b[0]]
        distance = Dis.min()
    else:
        sharpest_pair = [0,0]
        distance = 0.
    return sharpest_pair, distance

def random_subsample(neuron, num):
    """
    randomly selects a few nodes from neuron and builds a new neuron with them. The location of these node in the new neuron
    is the same as the original neuron and the morphology of them is such that if node A is parent (or grand parent) of node B
    in the original neuron, it is the same for the new neuron.

    Parameters
    ----------
    num: int
        number of nodes to be selected randomly.

    Returns
    -------
    Neuron: the subsampled neuron
    """

    I = np.arange(neuron.n_soma, neuron.n_node)
    np.random.shuffle(I)
    selected_index = I[0:num - 1]
    selected_index = np.union1d([0], selected_index)
    selected_index = selected_index.astype(int)
    selected_index = np.unique(np.sort(selected_index))

    return neuron_with_selected_nodes(neuron, selected_index)


def prune_subsample(neuron, number):
    main_point = subsample_main_nodes(neuron)
    Nodes = main_point.nodes_list
    rm = (main_point.n_node - number)/2.
    for remove in range(int(rm)):
        b, m = find_sharpest_fork(Nodes)
        remove_pair_adjust_parent(Nodes, b)

    return Neuron(input_format = 'only list of nodes', input_file = Nodes)

def neuron_with_axon(neuron):
    """
    Parameters:
    -----------
    neuron

    Return:
    -------
    neuron with axon (whitout dendrite)
    """
    (basal,) = np.where(neuron.nodes_type != 3)
    (apical,) = np.where(neuron.nodes_type != 4)
    selected = np.intersect1d(basal, apical)
    return neuron_with_selected_nodes(neuron, selected)

def neuron_with_dendrite(neuron):
    (axon,) = np.where(neuron.nodes_type != 2)
    return neuron_with_selected_nodes(neuron, axon)


def straight_subsample_with_fixed_number(neuron, num):
    """
    Returning a straightened subsample neuron with fixed number of nodes.

    Parameters
    ----------
    num: int
        number of nodes on the subsampled neuron

    Returns
    -------
    distance: float
        the subsampling distance
    neuron: Neuron
        the subsampled neuron
    """
    l = sum(neuron.distance_from_parent)
    branch_number = len(np.where(neuron.branch_order[neuron.n_soma:] == 2))
    distance = l/(num - branch_number)
    neuron = straigh_subsample(neuron, distance)
    return neuron, distance

# def parent_id(neuron, selected_index):
#     """
#     Return the parent id of all the selected_index of the neurons.

#     Parameters
#     ----------
#     selected_index: numpy array
#         the index of nodes

#     Returns
#     -------
#     parent_id: the index of parent of each element in selected_index in
#     this array.
#     """
#     length = len(neuron.nodes_list)
#     selected_length = len(selected_index)
#     adjacency = np.zeros([length,length])
#     adjacency[neuron.parent_index[1:], range(1,length)] = 1
#     full_adjacency = np.linalg.inv(np.eye(length) - adjacency)
#     selected_full_adjacency = full_adjacency[np.ix_(selected_index,selected_index)]
#     selected_adjacency = np.eye(selected_length) - np.linalg.inv(selected_full_adjacency)
#     selected_parent_id = np.argmax(selected_adjacency, axis=0)
#     return selected_parent_id

def prune(neuron,
          number_of_nodes,
          threshold):
    """
    Pruning the neuron. It removes all the segments that thier length is less
    than threshold unless the number of nodes becomes lower than lowest_number.
    In the former case, it removes the segments until the number of nodes is
    exactly the lowest_number.

    Parameters
    ----------
    neuron: Neuron
        input neuron.
    number_of_nodes: int
        the number of nodes for output neuron.

    Returns
    -------
    pruned_neuron: Neuron
        The pruned neuron.
    """
    n = len(neuron.nodes_list)
    for i in range(n - number_of_nodes):
        length, index = shortest_tips(neuron)
        if(length < threshold):
            neuron = remove_node(neuron, index)
        else:
            break
    neuron.set_distance_from_parent()
    return neuron

def shortest_tips(neuron):
    """
    Returing the initial node of segment with the given end point.
    The idea is to go up from the tip.
    """
    (endpoint_index,) = np.where(neuron.branch_order[neuron.n_soma:] == 0)
    (branch_index,) = np.where(neuron.branch_order[neuron.n_soma:] == 2)
    selected_index = np.union1d(neuron.n_soma + endpoint_index,
                                neuron.n_soma + branch_index)
    selected_index = np.append(0, selected_index)
    par = parent_id(neuron, range(1,len(endpoint_index) + 1))
    dist = neuron.location[:, endpoint_index] - neuron.location[:, par]
    lenght = sum(dist**2,2)
    index = np.argmin(lenght)

    return np.sqrt(min(lenght)), endpoint_index[index] + neuron.n_soma

def straight_prune_subsample(neuron, number_of_nodes):
    """
    Subsampling a neuron with straightening and pruning. At the first step, it
    strighten the neuron with 200 nodes (if the number of nodes for the
    neuron is less than 200, it doesn't change it). Then the neuron is pruned
    with a twice the distance used for straightening. If the number of nodes
    is less than 'number_of_nodes' the algorithm stops otherwise it increases
    the previous distance by one number and does the same on the neuron.

    Parameters
    ----------
    neuron: Neuron
        input neuron
    number_of_nodes: int
        the number of nodes for the output neuron

    Returns
    -------
    sp_neuron: Neuron
        the subsample neuron after straightening and pruning.
    """
    if(neuron.n_node > 200):
        neuron, distance = straight_subsample_with_fixed_number(neuron, 200)
    sp_neuron = prune(neuron=neuron, number_of_nodes=number_of_nodes, threshold=2*distance)
    while(len(sp_neuron.nodes_list)>number_of_nodes):
        distance += 1
        sp_neuron = straigh_subsample(sp_neuron, distance)
        sp_neuron = prune(neuron=sp_neuron,
                                 number_of_nodes=number_of_nodes,
                                 threshold=2*distance)
    return sp_neuron

def mesoscale_subsample(neuron, number):
    main_point = subsample_main_nodes(neuron)
    Nodes = main_point.nodes_list
    rm = (main_point.n_node - number)/2.
    for remove in range(int(rm)):
        b, m = find_sharpest_fork(neuron, Nodes)
        remove_pair_adjust_parent(neuron, Nodes, b)
    neuron = Neuron(input_format = 'only list of nodes', input_file = Nodes)

    if(neuron.n_node > number):
        (I,) = np.where(neuron.branch_order == 0)
        neuron = remove_node(neuron, I[0])

    return neuron

def subsample_main_nodes(neuron):
    """
    subsamples a neuron with its main node only; i.e endpoints and branching nodes.

    Returns
    -------
    Neuron: the subsampled neuron
    """
    # select all the main points
    selected_index = get_index_of_main_nodes(neuron)

    # Computing the parent id of the selected nodes
    n = neuron_with_selected_nodes(neuron, selected_index)
    return n

def find_sharpest_fork(neuron, Nodes):
    """
    Looks at the all branching point in the Nodes list, selects those which both its children are end points and finds
    the closest pair of childern (the distance between children).
    Parameters
    ----------
    Nodes: list
    the list of Node

    Returns
    -------
    sharpest_pair: array
        the index of the pair of closest pair of childern
    distance: float
        Distance of the pair of children
    """
    pair_list = []
    Dis = np.array([])
    for n in Nodes:
        if n.parent is not None:
            if n.parent.parent is not None:
                a = n.parent.children
                if(isinstance(a, list)):
                    if(len(a)==2):
                        n1 = a[0]
                        n2 = a[1]
                        if(len(n1.children) == 0 and len(n2.children) == 0):
                            pair_list.append([n1 , n2])
                            dis = LA.norm(a[0].xyz - a[1].xyz,2)
                            Dis = np.append(Dis,dis)
    if(len(Dis)!= 0):
        (b,) = np.where(Dis == Dis.min())
        sharpest_pair = pair_list[b[0]]
        distance = Dis.min()
    else:
        sharpest_pair = [0,0]
        distance = 0.
    return sharpest_pair, distance

def find_sharpest_fork_general(neuron, Nodes):
    """
    Looks at the all branching point in the Nodes list, selects those which both its children are end points and finds
    the closest pair of childern (the distance between children).
    Parameters
    ----------
    Nodes: list
    the list of Node

    Returns
    -------
    sharpest_pair: array
        the index of the pair of closest pair of childern
    distance: float
        Distance of the pair of children
    """
    pair_list = []
    Dis = np.array([])
    for n in Nodes:
        if n.parent is not None:
            if n.parent.parent is not None:
                a = n.parent.children
                if(isinstance(a, list)):
                    if(len(a)==2):
                        n1 = a[0]
                        n2 = a[1]
                        pair_list.append([n1 , n2])
                        dis = LA.norm(a[0].xyz - a[1].xyz,2)
                        Dis = np.append(Dis,dis)
    if(len(Dis)!= 0):
        (b,) = np.where(Dis == Dis.min())
        sharpest_pair = pair_list[b[0]]
        distance = Dis.min()
    else:
        sharpest_pair = [0,0]
        distance = 0.
    return sharpest_pair, distance

def remove_pair_replace_node(neuron, Nodes, pair):
    """
    Removes the pair of nodes and replace it with a new node. the parent of new node is the parent of the pair of node,
    and its location and its radius are the mean of removed nodes.
    Parameters
    ----------
    Nodes: list
    the list of Nodes

    pair: array
    The index of pair of nodes. the nodes should be end points and have the same parent.

    Returns
    -------
    The new list of Nodes which the pair are removed and a mean node is replaced.
    """

    par = pair[0].parent
    loc = pair[0].xyz + pair[1].xyz
    loc = loc/2
    r = pair[0].r + pair[1].r
    r = r/2
    Nodes.remove(pair[1])
    Nodes.remove(pair[0])
    n = McNeuron.Node()
    n.xyz = loc
    n.r = r
    par.children = []
    par.add_child(n)
    n.parent = par
    Nodes.append(n)

def remove_pair_adjust_parent(neuron, Nodes, pair):
    """
    Removes the pair of nodes and adjust its parent. the location of the parent is the mean of the locaton of two nodes.

    Parameters
    ----------
    Nodes: list
    the list of Nodes

    pair: array
    The index of pair of nodes. the nodes should be end points and have the same parent.

    Returns
    -------
    The new list of Nodes which the pair are removed their parent is adjusted.
    """

    par = pair[0].parent
    loc = pair[0].xyz + pair[1].xyz
    loc = loc/2
    Nodes.remove(pair[1])
    Nodes.remove(pair[0])
    par.xyz = loc
    par.children = []

def parent_id_for_extract(original_parent_id, selected_index):
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
    length = len(original_parent_id)
    selected_length = len(selected_index)
    adjacency = np.zeros([length, length])
    adjacency[original_parent_id[1:]-1, range(1, length)] = 1
    full_adjacency = np.linalg.inv(np.eye(length) - adjacency)
    selected_full_adjacency = full_adjacency[np.ix_(selected_index, selected_index)]
    selected_adjacency = np.eye(selected_length) - np.linalg.inv(selected_full_adjacency)
    selected_parent_id = np.argmax(selected_adjacency, axis=0)
    return selected_parent_id

def parent_id_for_extract2(original_parent_id, selected_index):
    parent_id = np.array([], dtype=int)
    for i in selected_index[1:]:
        p = original_parent_id[i]
        while(~np.any(selected_index == p-1)):
            p = original_parent_id[p-1]
        (ind,) = np.where(selected_index == p-1)
        parent_id = np.append(parent_id, ind)
    parent_id = np.append(-1, parent_id)
    return parent_id

def extract_main_neuron_from_swc(matrix, num = 300):
    a, b = np.unique(matrix[:,6],return_counts=True)
    (I,) = np.where(b==2)
    branch_point = a[I]
    end_point = np.setxor1d(np.arange(0,matrix.shape[0]), matrix[4:,6])
    I = np.union1d(branch_point, end_point)
    leng = matrix.shape[0]
    lengi = len(I)
    I = np.union1d(I, np.arange(0,n, int(n/(leng - lengi - 1))))
    random_point = np.setxor1d(np.arange(3,matrix.shape[0]), I)
    I = np.append(I, random_point[-(num - len(I)):-1])
    I = np.sort(I)
    I = np.array(I,dtype=int)
    I[0] = 0
    K = matrix[:,6]
    K = np.array(K,dtype=int)
    J = parent_id_for_extract2(K, I)

    J = J + 1
    J[0] = -1
    n = len(J)

    I = I - 1
    I = I[1:]
    I = np.append(I,matrix.shape[0]-1)
    new_matrix = np.zeros([n, 7])
    new_matrix[:,0] = np.arange(0,n)
    new_matrix[:,1] = matrix[I,1]
    new_matrix[:,2] = matrix[I,2]
    new_matrix[:,3] = matrix[I,3]
    new_matrix[:,4] = matrix[I,4]
    new_matrix[:,5] = matrix[I,5]
    new_matrix[:,6] = J
    neuron = Neuron(input_format='Matrix of swc', input_file=new_matrix)
    return neuron

def rescale_neuron_in_unit_box(neuron):
    swc = np.zeros([neuron.n_node, 7])
    swc[:, 1] = neuron.diameter
    ratio = np.max(np.abs(neuron.location))
    swc[:, 2] = rescale_location(neuron.location[0, :])
    swc[:, 3] = rescale_location(neuron.location[1, :])
    swc[:, 4] = rescale_location(neuron.location[2, :])
    swc[:, 5] = neuron.diameter
    swc[:, 6] = neuron.parent_index + 1
    swc[0, 6] = -1
    return swc

def rescale_location(loc, ratio):
    loc = loc/ratio
    loc = loc - loc[0]
    return loc

def cut_from_end_nodes(neuron, number):
    current_swc = rescale_neuron_in_unit_box(neuron)
    current_neuron = Neuron(input_file=current_swc, input_format="Matrix of swc")
    current_swc = McNeuron.swc_util.get_swc_matrix(current_neuron)
    for i in range(current_neuron.n_node - number):
        row = np.random.choice(np.where(current_neuron.branch_order == 0)[0])
        current_swc = np.delete(current_swc, row, axis=0)
        higher_index = np.where(current_swc[:,6]>row)[0]
        current_swc[higher_index, 6] = current_swc[higher_index, 6] - 1
        current_neuron = Neuron(input_file=current_swc,
                                         input_format="Matrix of swc")
    current_swc = rescale_neuron_in_unit_box(current_neuron)
    current_neuron = Neuron(input_file=current_swc,
                                     input_format="Matrix of swc")
    return current_neuron

def _get_index_of_main_nodes(n_soma, original_parent):
    n_node = original_parent.shape[0]
    branch_order = np.zeros(n_node)
    unique, counts = np.unique(original_parent[1:], return_counts=True)
    branch_order[unique.astype(int)] = counts
    # if(len(np.where(branch_order>2)[0])>1):
    #     print("Error: Neuron is not binary")

    (branch_index,) = np.where(branch_order[n_soma:] == 2)
    (endpoint_index,) = np.where(branch_order[n_soma:] == 0)
    selected_index = np.union1d(branch_index + n_soma,
                                endpoint_index + n_soma)
    # selected_index = np.append(range(n_soma), selected_index)
    selected_index = np.append(0, selected_index)
    return selected_index


def _get_index_node_straighten(index_of_main_nodes, original_parent, M, length):
    distance_from_parent = np.sqrt(((M[:, 2:5] - M[original_parent, 2:5])**2).sum(axis=1))
    selected_index = np.array([])
    n_selected_node_on_each_segment = np.zeros(len(index_of_main_nodes)-1)
    for c in range(len(index_of_main_nodes)-1):
        seq = np.arange(index_of_main_nodes[c]+1, index_of_main_nodes[c+1]+1)
        if(len(seq) != 0):
            (subseq,) = np.where(np.diff(np.floor(np.cumsum(distance_from_parent[seq])/length)))
            n_selected_node_on_each_segment[c] = len(subseq)
            selected_index = np.append(selected_index, index_of_main_nodes[c] + 1 + np.array(subseq))
        else:
            n_selected_node_on_each_segment[c] = 0

    return n_selected_node_on_each_segment, selected_index.astype(int)

def _get_parent_main(original_parent, index_of_main_nodes):
    parsub = original_parent[index_of_main_nodes[:-1]+1]
    index = 0
    ppp = np.zeros(len(index_of_main_nodes))
    for i in index_of_main_nodes[1:]:
        (pp,) = np.where(index_of_main_nodes==parsub[index])
        index +=1
        ppp[index] = pp
    return ppp

def check_standard_format(swc_matrix):
    original_parent = swc_matrix[:, 6].astype(int)-1
    original_parent[0] = 0
    n_soma = len(np.where(swc_matrix[:,1] == 1)[0])
    index_of_main_nodes = _get_index_of_main_nodes(n_soma, original_parent).astype(int)
    parsub = original_parent[index_of_main_nodes[:-1]+1]
    index = 0
    ppp = np.zeros(len(index_of_main_nodes))
    for i in index_of_main_nodes[1:]:
        (pp,) = np.where(index_of_main_nodes==parsub[index])
        index +=1
        if(len(pp)!=1):
            print("Error: Not Standard")

def remove_tip_for_3forks(swc_matrix):
    a = swc_matrix
    for i in range(2,swc_matrix.shape[0]):
        (I,) = np.where(a[:,6]==i)
        if(len(I)>2):
            for j in I:
                (J,) = np.where(a[:,6]==j)
                if(len(J)==0):
                    a = remove_one_tip_from_swc_file(a, j)
    return a

def remove_one_tip_from_swc_file(swc_matrix, node_index):
    a = np.zeros([swc_matrix.shape[0]-1, 7])
    a[:node_index,:] = swc_matrix[:node_index,:]
    a[node_index:,:] = swc_matrix[node_index+1:,:]
    (I,) = np.where(a[:,6]>node_index)
    a[I,6] = a[I,6]-1
    return a

def make_dense_morphology(swc, n_conse = 10):
    """
    Returns:
    dense_locations: 3d locations of the nodes
    """
    location = swc[:,2:5]
    n_node = swc.shape[0]
    parent_index = swc[:,6]
    parent_index -= 1
    parent_index[0]=0
    a = location - location[parent_index.astype(int),:]
    dense_locations = np.zeros([n_node*n_conse,3])
    for i in range(n_conse):
        dense_locations[n_node*i:n_node*(i+1),:] = \
            location[parent_index.astype(int),:] + (float(i)/float(n_conse))*a
    return dense_locations

def pic(location, mesh=10, x_min=-200, x_max=200,y_min=-200,y_max=200):
    """
    Calculate the number of nodes in a 2d equidistance mesh.
    """
    image = np.zeros([mesh , mesh])
    x_all = location[:, 0]
    y_all = location[:, 1]
    index = np.logical_and(np.logical_and(x_all>x_min, x_all<x_max),
                          np.logical_and(y_all>y_min, y_all<y_max))
    x = (x_all[index]-x_min)/(x_max-x_min)
    x = np.floor((mesh-1)*x)
    y = (y_all[index]-y_min)/(y_max-y_min)
    y = np.floor((mesh-1)*y)
    image[x.astype(int),y.astype(int)] = 1
    return image