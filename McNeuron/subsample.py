"""Collection of subsampling method on the neurons."""
import numpy as np
from numpy import linalg as LA
from McNeuron import tree_util
from copy import deepcopy
from sklearn import preprocessing

def subsample_swc(swc_matrix,
              subsample_type='nothing',
              length = 1):
    """
    it first decomposes the tree structure of the neuron by
    breaking it to its segments and the topological tree structures.
    (Having both these components, we can assemble the tree again.)
    Then, by going into each segments, the subsampling methodsis applied.


    Parameters:
    -----------
    swc_matrix: a numpy array of shape [n, 7]
        n is representing the number of node that describe the neuron.

    subsample_type: str
        the type of subsampling. It can be one of these methods:
        'nothing': doesn't change anything
        'regular': just preserves the end points and the branching points
        'straigthen': straigthens the neuron based on the length. it approximate
        the segments such that the distance between two consequative nodes is around
        'length'. For more info check the function 'straigthen_segment'.
    length: float
        the value to subsample the segment

    Returns:
    --------
    swc_matrix: a numpy array of shape [n, 7]
        the subsampled neuron

    """
    list_all_segment, tree = list_of_segments(swc_matrix)
    return subsample_with_segment_list(list_all_segment,
                                tree,
                                subsample_type='nothing',
                                length = 1)

def subsample_with_segment_list(list_all_segment, tree, subsample_type, length):
    """
    Given the list of segments and the tree structure, this function assemble them based
    on the subsample type


    Parameters:
    -----------
    list_all_segment: list of numpy arraies of shape [n, 7]
        each element of the list is a swx matix of the indexed segment. Then segments
        are connected by the binary graph of 'tree'.

    tree: numpy array of integers
        the parent id of the binary graph that shows the neuron

    subsample_type: str
        the type of subsampling. It can be one of these methods:
        'nothing': doesn't change anything
        'regular': just preserves the end points and the branching points
        'straigthen': straigthens the neuron based on the length. it approximate
        the segments such that the distance between two consequative nodes is around
        'length'. For more info check the function 'straigthen_segment'.
    length: float
        the value to subsample the segment

    Returns:
    --------
    swc_matrix: a numpy array of shape [n, 7]
        the subsampled neuron

    """
    swc_matrix = subsample_segment(list_all_segment[0],
                                   subsample_type=subsample_type,
                                   length=length)[1:,:]
    last_index_segment = np.array([len(list_all_segment[0])-1])
    for i in range(1, len(tree)):
        segment = subsample_segment(list_all_segment[i],
                                   subsample_type=subsample_type,
                                   length=length)[1:,:]
        segment[0, 6] = last_index_segment[tree[i]]
        n_till_now = swc_matrix.shape[0]
        n_will_be = segment.shape[0]
        segment[1:, 6] = range(n_till_now+1, n_till_now + n_will_be)
        swc_matrix = np.append(swc_matrix, segment, axis=0)
        last_index_segment = np.append(last_index_segment, n_till_now + n_will_be)
    return swc_matrix

def list_of_segments(swc_matrix):
    """
    It decompose the tree to its segments and return the list of all segments
    and their tree structure.

    Parameters:
    -----------
    swc_matrix: a numpy array of shape [n, 7]
        swc rep of neuron.

    Returns:
    --------
    list_all_segment: list of numpy arraies of shape [n, 7]
        each element of the list is a swx matix of the indexed segment. Then segments
        are connected by the binary graph of 'tree'.

    tree: numpy array of integers
        the parent id of the binary graph that shows the neuron
    """
    swc_matrix = deepcopy(swc_matrix)
    parent_index = tree_util.get_parent_index(swc_matrix)
    main_index = tree_util.get_index_of_critical_points(swc_matrix,
                                                   input_type='swc_matrix',
                                                   only_one_somatic_node=False)
    list_all_segment = []
    segment_name = np.array([])
    segment_parent_name = np.array([])
    for node in main_index:
        segment = np.array([node])
        segment_name = np.append(segment_name, node)
        current_node = parent_index[node]
        while current_node not in main_index:
            segment = np.append(current_node, segment)
            current_node = parent_index[current_node]
        segment = np.append(current_node, segment)
        segment_parent_name = np.append(segment_parent_name, current_node)
        parent_segment = current_node
        segment_swc = swc_matrix[segment,:]
        segment_swc[:,6] = range(segment_swc.shape[0])
        list_all_segment.append(segment_swc)
    le = preprocessing.LabelEncoder()
    le.fit(segment_name)
    tree = le.transform(segment_parent_name)
    return list_all_segment, tree

def assemble_segments(list_all_segment, tree):
    """
    It attached the segments by tree structure

    Parameters:
    -----------
    list_all_segment: list of numpy arraies of shape [n, 7]
        each element of the list is a swx matix of the indexed segment. Then segments
        are connected by the binary graph of 'tree'.

    tree: numpy array of integers
        the parent id of the binary graph that shows the neuron

    Returns:
    --------
    swc_matrix: a numpy array of shape [n, 7]
        swc rep of neuron.
    """
    swc_matrix = list_all_segment[0][1:,:]
    name_segment = np.array([len(list_all_segment[0])-1])
    for i in range(1, len(tree)):
        segment = list_all_segment[i]
        segment[0, 6] = name_segment[tree[i]]
        n_till_now = swc_matrix.shape[0]
        n_will_be = segment.shape[0]
        segment[1:, 6] = range(n_till_now+1, n_till_now + n_will_be)
        swc_matrix = np.append(swc_matrix, segment, axis=0)
        name_segment = np.append(name_segment, n_till_now + n_will_be)
    return swc_matrix

def straigthen_segment(segment, length):
    """

    Parameters:
    -----------
    segment: a numpy array of shape [n, 7]
        n is representing the number of node that describe the segments and
        should be at least 7.

    length: float
        the value to subsample the segment

    Returns:
    --------
    swc_matrix: the subsampled segments
        it approximates the segment by the length

    """
    swc_matrix = segment
    distance_from_parent_2 = (swc_matrix[:-1, 2:5] - swc_matrix[1:, 2:5]) ** 2
    distance_from_parent = np.sqrt(distance_from_parent_2.sum(axis=1))

    distance_from_root = np.cumsum(distance_from_parent)
    distance_from_root = np.append(0, distance_from_root)
    len_seg = distance_from_root[-1]
    n = int(len_seg/length)+1

    propose_dis =  (len_seg/(n+1))*np.arange(n+1)

    upper_index = np.searchsorted(distance_from_root, propose_dis)
    #upper_index[-1] -=1
    lower_index =  upper_index - 1
    lower_index[0] = 0
    upper_index[0] = 1

    weigth_lower = propose_dis - distance_from_root[lower_index]
    weigth_upper = distance_from_root[upper_index] - propose_dis
    total_weigth = weigth_lower + weigth_upper
    overlap_points = total_weigth==0
    total_weigth[overlap_points] = 1
    weigth_lower[overlap_points] = 1
    weigth_upper, weigth_lower = weigth_lower/(total_weigth), weigth_upper/(total_weigth)
    weigth_upper = np.expand_dims(weigth_upper, 1)
    weigth_lower = np.expand_dims(weigth_lower, 1)
    new_swc = np.zeros([n+1,7])
    new_swc[:,2:6] = weigth_lower * swc_matrix[lower_index, 2:6] + \
                     weigth_upper * swc_matrix[upper_index, 2:6]
    new_swc[:,1] = swc_matrix[upper_index, 1]
    new_swc[:,6] = range(n+1)
    return new_swc

def subsample_segment(swc_matrix,
                     subsample_type='regular',
                     length = 1):
    if subsample_type =='nothing':
        return deepcopy(swc_matrix)
    if subsample_type =='regular':
        return deepcopy(swc_matrix[[0,-1],:])
    if subsample_type =='straigthen':
        return straigthen_segment(swc_matrix, length)

class Subsample(object):

    def set_swc(self, swc_matrix):
        self.swc_matrix = swc_matrix

    def fit(self):
        """ set the setments and tree"""
        self.list_all_segment, self.tree = list_of_segments(self.swc_matrix)

    def subsample(self, subsample_type='nothing', length = 1):
        """
        Parameters:
        -----------
        subsample_type: str
            the type of subsample. It can be:
            'nothing': doesn't change anything
            'regular': just preserves the end points and the branching points
            'straigthen': straigthens the neuron based on the length. it approximate
            the segments such that the distance between two consequative nodes is around
            'length'. For more info check the function 'straigthen_segment'.

        length: float
            the value to subsample the segment

        Returns:
        --------
        swc_matrix: the subsampled segments
            it approximates the segment by the length

        """
        return subsample_with_segment_list(list_all_segment=self.list_all_segment,
                                           tree=self.tree,
                                           subsample_type=subsample_type,
                                           length=length)
