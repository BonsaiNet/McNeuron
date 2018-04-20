"""Neuron class to extract the features of the neuron and perturb it."""
import numpy as np
from numpy import linalg as LA
import math
from scipy.sparse import csr_matrix
from __builtin__ import str
from copy import deepcopy
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn import preprocessing
from McNeuron import swc_util
from McNeuron import tree_util

#np.random.seed(0)

class Neuron(object):
    def __init__(self, input_file=None):
        """
        Default constructor.
        Parameters
        -----------
            
        n_soma : int
            The number of the nodes that represents the soma.
        n_node : int
            The number of all the nodes in the neuron.
        location : array of shape = [3, n_node]
            Three dimentional location of the nodes.
        parent_index : array of integers
            The index of the parent for each node of the tree. It starts from 0 and the parent
            of the root is 0.
        """
        if isinstance(input_file, np.ndarray):
            swc_matrix = deepcopy(input_file)
        else:
            swc_matrix = np.loadtxt(input_file)
        swc_matrix = swc_util.correct_swc(swc_matrix)
        self.read_swc_matrix(swc_matrix)   
        self.features = {}
        #self.set_features()

    def __str__(self):
        """
        describtion.
        """
        return "Neuron found with " + str(self.n_node) + " number of nodes and"+ str(self.n_soma) + "number of node representing soma."

    def set_features(self):
        self.basic_features()
        #self.motif_features()
        self.geometrical_features()
        #self.ect_features()

    def basic_features(self):
        """
        Returns:
        --------
            branch_order : array of shape = [n_node]
            The number of children of the nodes. It can be and integer number for
            the root (first element) and only 0, 1 or 2 for other nodes.
        """
        branch_order = tree_util.branch_order(self.parent_index)
        self.features['branch order'] = branch_order
        self.features['Nnodes'] = np.array([self.n_node - self.n_soma])
        self.features['Nsoma'] = np.array([self.n_soma])
        (num_branches,) = np.where(branch_order[self.n_soma:] >= 2)
        self.features['Nbranch'] = np.array([len(num_branches)])
        (num_pass,) = np.where(branch_order[self.n_soma:] == 1)
        self.features['Npassnode'] = np.array([len(num_pass)])
        self.features['initial segments'] = np.array([branch_order[0]])
        
    def motif_features(self):
        branch_order = tree_util.branch_order(self.parent_index)
        distance_from_parent = self.distance_from_parent()
        
        main, parent_main_point, neural, euclidean = \
            self.get_neural_and_euclid_lenght_main_point(branch_order, distance_from_parent)
            
        branch_branch, branch_die, die_die, initial_with_branch = \
            self.branching_type(main, parent_main_point)

        branch_depth, continue_depth, dead_depth, branch_branch_depth, branch_die_depth, die_die_depth = \
            self.type_at_depth(branch_order, self.parent_index, branch_branch, branch_die, die_die)             
        main_parent = self.get_parent_index_of_subset(main, parent_main_point)    
        
        main_branch_depth, _, main_dead_depth, _, _, _ = \
            self.type_at_depth(tree_util.branch_order(main_parent), main_parent)   
        self.features['branch depth'] = branch_depth
        self.features['continue depth'] = continue_depth
        self.features['dead depth'] = dead_depth
        self.features['main branch depth'] = main_branch_depth
        self.features['main dead depth'] = main_dead_depth
        self.features['branch branch'] = np.array([len(branch_branch)])
        self.features['branch die'] = np.array([len(branch_die)])
        self.features['die die'] = np.array([len(die_die)])
        self.features['branch branch depth'] = branch_branch_depth
        self.features['branch die depth'] = branch_die_depth
        self.features['die die depth'] = die_die_depth
        self.features['initial with branch'] = np.array([initial_with_branch])
        self.features['all non trivial initials'] = np.array([self.all_non_trivial_initials()])
        a = float(self.features['die die']*self.features['branch branch'])
        asym = 0
        if a != 0:
            asym = float((4*self.features['branch die']**2))/a
        self.features['asymmetric ratio'] = np.array([asym]) 
        
    def geometrical_features(self):
        """
        Given:
            n_node
            n_soma
            branch_order
            nodes_diameter
            parent_index
            location

        ext_red_list : array of shape = [3, n_node]
            first row: end points and order one nodes (for extension)
            second row: end points (for removing)
            third row: end point wich thier parents are order one nodes
            (for extension)
        connection :  array of shape = [n_node, n_node]
            The matrix of connectivity of the nodes. The element (i,j) of the
            matrix is not np.nan if node i is a decendent of node j. The value at
            this array is the distance of j to its parent. It's useful for the
            calculation of the neural distance over Euclidain distance.
        frustum : array of shape = [n_node] !!!NOT IMPLEMENTED!!!
            The value of th fustum from the node toward its parent.
        branch_order : array of shape = [n_node]
            The number of children for each of nodes. Notice that for the nodes
            rather than root it should be 0, 1 or 2. For root it can be any integer
            number.
        rall_ratio :  array of shape = [n_node] !!!NOT IMPLEMENTED!!!
            It's not nan only in branching nodes which its value is the rall ratio.
        distance_from_root : array of shape = [n_node]
            Euclidain distance toward the root.
        distance_from_parent : array of shape = [n_node]
            Euclidain distance toward the parent of the node.
        slope : array of shape = [n_node]
            ratio of euclidain distance toward the parent of the node over their
            diameter difference.
        branch_angle : array of shape [3, n_nodes]
            it shows the angles at the branching nodes: First row is the angle of
            two outward segments at the branching point Second and third rows are
            the angle betwen two outward segments and previous segment at the
            branching in arbitrary order (nan at other nodes).
        angle_global : array of shape = [n_node]
            The angle between the line linking the node to the root and the line
            likning it to its parent.
        local_angle : array of shape = [n_node]
            The angle between the line linking the node to its parent and its child
            and nan otherwise.
        """
        branch_order = tree_util.branch_order(self.parent_index)
        distance_from_parent = self.distance_from_parent()
        mean_distance_from_parent = distance_from_parent.mean()
        distance_from_root = self.distance_from_root()
        neural_all = self.neural_distance_from_root(distance_from_parent)
        local_angle = self.local_angle(branch_order)
        global_angle = self.global_angle()
        branch_angle, side_angle = self.branch_angle(branch_order)
        curvature = self.set_curvature()
        fractal = self.set_discrepancy(np.arange(.01, 5,.01))
        main, parent_main_point, neural, euclidean = \
            self.get_neural_and_euclid_lenght_main_point(branch_order, distance_from_parent)

        self.features['global angle'] = global_angle[self.n_soma:]
        self.features['local angle'] = local_angle[self.n_soma:]
        self.features['distance from parent'] = distance_from_parent[self.n_soma:]
        self.features['distance from root'] = distance_from_root[self.n_soma:]/mean_distance_from_parent
        self.features['neuronal/euclidean'] = neural_all[self.n_soma:]/distance_from_root[self.n_soma:]
        self.features['mean bending'] = \
           np.array([((self.features['neuronal/euclidean'] - 1.)).mean()])
        self.features['branch angle'] = branch_angle
        self.features['side branch angle'] = side_angle
        #self.features['curvature'] = curvature
        self.features['discrepancy space'] = fractal[:,0]
        self.features['self avoidance'] = fractal[:,1]
        self.features['pictural image xy'] = self.set_pictural_xy()
        #self.features['pictural image xyz'] = self.set_pictural_xyz(5., 5., 5.)
        #self.features['cylindrical density'] = 
        #self.features['pictural image xy tips'] = self.set_pictural_tips(branch_order,10., 10.)
        self.features['segmental neural length'] = neural/mean_distance_from_parent
        self.features['segmental euclidean length'] = euclidean/mean_distance_from_parent
        self.features['mean segmental neural length'] = \
           np.array([self.features['segmental neural length'].mean()])
        self.features['mean segmental euclidean length'] = \
           np.array([self.features['segmental euclidean length'].mean()])
        self.features['neuronal/euclidean for segments'] = neural/euclidean
        self.features['mean segmental neuronal/euclidean'] = \
           np.array([np.sqrt(((self.features['neuronal/euclidean for segments'] - 1.)**2).mean())])
        self.features['segmental branch angle'] = \
           self.set_branch_angle_segment(main, parent_main_point)
    
    def set_branch_order(self):
        if 'branch order' not in self.features.keys():
            self.features['branch order'] = tree_util.branch_order(self.parent_index)
        
    def diameter_features(self):
        distance_from_root = self.distance_from_root()
        diameter_euclidean = self.diameter_euclidean(distance_from_root, bins=10)
        self.features['diameter euclidean (bins)'] = diameter_euclidean
    
    def ect_features(self):
        self.features['toy'] = \
           np.array([np.sqrt((self.features['neuronal/euclidean for segments']).mean())])
        
    def get_index_main_nodes(self):
        """
        Returing the index of end points and branching points.

        Returns
        -------
        important_node: numpy array
            the index of main points.
        """
        (branch_index, ) = np.where(tree_util.branch_order[self.n_soma:] == 2)
        (end_nodes, ) = np.where(tree_util.branch_order[self.n_soma:] == 0)
        important_node = np.append(branch_index, end_nodes)
        if(len(important_node) != 0):
            important_node = self.n_soma + important_node
        return important_node

    def neural_distance_from_root(self, distance_from_parent):
        a = np.arange(self.n_node)
        dis = np.zeros(self.n_node)
        while(sum(a) != 0):
            dis += distance_from_parent[a]
            a = self.parent_index[a]
        return dis

    def get_neural_and_euclid_lenght_main_point(self, branch_order, distance_from_parent):
        (main_node,) = np.where(branch_order[1:] != 1)
        main_node += 1
        main_node = np.append(0, main_node)

        parent_main_node = np.arange(self.n_node)

        neural = np.zeros(self.n_node)
        for i in range(1, self.n_node):
            if(branch_order[i] == 1):
                neural[i] = neural[self.parent_index[i]] + distance_from_parent[i]
                parent_main_node[i] = parent_main_node[self.parent_index[i]]
        neural = neural[self.parent_index[main_node]][1:] + distance_from_parent[main_node][1:]
        parent_main_node = parent_main_node[self.parent_index[main_node]]
        euclidean = LA.norm(self.location[:, main_node] -
                            self.location[:, parent_main_node], axis=0)[1:]
        return main_node, parent_main_node, neural, euclidean


    def get_parent_index_of_subset(self, subset, parent_of_subset):
        le = preprocessing.LabelEncoder()
        le.fit(subset)
        parent = le.transform(parent_of_subset)
        return parent
    
    def branching_type(self, main_node, parent_main_node):
        le1 = preprocessing.LabelEncoder()
        le2 = preprocessing.LabelEncoder()
        le1.fit(main_node)
        parent_main_node_transform = le1.transform(parent_main_node)
        branch_main_point = np.unique(parent_main_node_transform)
        parent_branch_main_point = parent_main_node_transform[branch_main_point]
        le2.fit(branch_main_point)
        parent_branch_main_point_transform = le2.transform(parent_branch_main_point)
        branch_order_main_point = tree_util.branch_order(parent_branch_main_point_transform)
        branch_branch = np.where(branch_order_main_point[1:] == 2)[0] + 1
        branch_die = np.where(branch_order_main_point[1:] == 1)[0] + 1
        branch_branch = le1.inverse_transform(le2.inverse_transform(branch_branch))
        branch_die = le1.inverse_transform(le2.inverse_transform(branch_die))
        die_die = np.setxor1d(np.setxor1d(np.unique(parent_main_node)[1:], branch_branch), branch_die)
        initial_with_branch = branch_order_main_point[0] - 1
        return branch_branch, branch_die, die_die, initial_with_branch

    def all_non_trivial_initials(self):
        (I,)=np.where(self.parent_index==0)
        count=0
        for i in I:
            if i!=0:
                (J,)=np.where(self.parent_index == i)
                if(len(J) != 0):
                    count += 1
        return count

    def make_fixed_length_vec(self, input_vec, length_vec):
        l = len(input_vec)
        if(l < length_vec):
            fixed_vec = np.zeros(length_vec)
            fixed_vec[0:l] = input_vec
        else:
            fixed_vec = input_vec[:length_vec]
        return fixed_vec

    def get_neural_and_euclid_lenght_old_method(self, initial_index_of_node,thier_parents):
        """
        Returning the neural and Euclidain length for the given points.
        """
        neural = np.array([])
        euclidan = np.array([])
        for i in range(initial_index_of_node.shape[0]):
            neural_length = self.distance(initial_index_of_node[i],
                                          thier_parents[i])
            euclidan_length = \
                LA.norm(self.location[:, initial_index_of_node[i]] -
                        self.location[:, thier_parents[i]], 2)
            neural = np.append(neural, neural_length)
            euclidan = np.append(euclidan, euclidan_length)
        return neural, euclidan

    def imagary(self, bins=10, projection=[0,1,2], nodes_type='all'):
        """
        Imaging neuron on a lattice of given size.

        Parameters:
        -----------
        bins: int or list
            if int, it is the number of bins that should be used for axes.
            if list, the bins corrosponds to the projection axes.
        projection: list
            The range of axes to project neuron's location. 0 ,1 ,2 are corresponding
            to 'x', 'y' and 'z'.
        nodes_type: str
            it can be 'all', 'tips', 'branches' and 'tips and branches'

        Returns:
        --------
        image: numpy 
            An array that shows the density of neuron in different locations.

        """
        dim = len(projection)
        if isinstance(bins, int):
            image = np.zeros(dim*[bins])
            bins = np.array([dim*[bins]]).T
        else:
            image = np.zeros(bins)
            bins = np.array([bins]).T
        location = self.projection_tips(projection=projection,
                                        nodes_type=nodes_type)
        normalized_locations = \
            np.floor((bins-1)*swc_util.map_to_zero_one(vector=location))
        index, count = np.unique(normalized_locations.astype(int), return_counts=True, axis=1)

        if dim == 1:
            image[index] = count
        elif dim == 2:
            image[index[0,:], index[1,:]] = count
            image = np.flipud(image.T)
        elif dim==3:
            image[index[0,:], index[1,:], index[2,:]] = count
        return image

    def discrepancy(self, box_length, projection=[0,1,2], nodes_type='all'):
        """
        Counting the number of occupied boxes, when the location of neuron is approximated by
        equal boxes

        Parameters:
        -----------
        box_length: list (or numpy) of positive numbers
            the list of all the lengths that are going to use for the approximation. 
            Notice: if there is only one length, wrap it in the bracket: e.g. box_length=[5]
        projection: list
            The range of axes to project neuron's location. 0 ,1 ,2 are corresponding
            to 'x', 'y' and 'z'.
        nodes_type: str
            it can be 'all', 'tips', 'branches' and 'tips and branches' 

        Return:
        -------
        precentage_list: numpy
            the percentage of occupied boxes relative to the whole number of nodes
            that neuron has in the given box_length.
        """ 
        number_of_nodes = [self.boxing(box_length=float(box_length[i])*np.ones([3,1]), 
                                       projection=projection,
                                       nodes_type=nodes_type).shape[1] \
                   for i in range(len(box_length))]
        precentage_list = np.array(number_of_nodes)/float(self.n_node)
        return precentage_list

    def boxing(self, 
               box_length, 
               return_counts=False, 
               projection=[0,1,2],
               nodes_type='all'):
        """
        Approximating the location of a neuron by equal sized boxes.

        Parameters:
        -----------
        box_length: float > 0
            the length of box.
        return_counts: boolean
            If true, it returns the number of nodes in the given box.
        projection: list
            The range of axes to project neuron's location. 0 ,1 ,2 are corresponding
            to 'x', 'y' and 'z'.
        nodes_type: str
            it can be 'all', 'tips', 'branches' and 'tips and branches' 

        Returns:
        --------
        box: numpy
            the location of the boxes that contain at least one node. If return_counts is True,
            it is a tuple that the first component is the location and the second component
            is the number of nodes in the boxes
        """
        locations = self.projection_tips(projection=projection, 
                                         nodes_type=nodes_type)    
        locations = np.floor(locations/box_length + .5)*box_length
        box = np.unique(locations, return_counts=return_counts, axis=1)
        return box

    def projection_tips(self, projection, nodes_type='all'):
        """
        Retrning the location of the neuron (or its tips) for given projection.

        Parameters:
        -----------
        projection: list
            The range of axes to project neuron's location. 0 ,1 ,2 are corresponding
            to 'x', 'y' and 'z'.
        nodes_type: str
            it can be 'all', 'tips', 'branches' and 'tips and branches' 

        """
        self.set_branch_order()
        if nodes_type=='all':
            locations = self.location[projection, :]
        else:
            self.set_branch_order()
            if nodes_type=='tips':
                (nodes,) = np.where(self.features['branch order']==0)
            elif nodes_type=='branches':
                (nodes,) = np.where(self.features['branch order']==2)
            elif nodes_type=='tips and branches' :
                (nodes,) = np.where(self.features['branch order']!=1)
            locations = self.location[np.ix_(projection, nodes)]
        return locations

    def repellent(self, box_length, nodes_type='all'):
        """
        Returning frequency of the number boxes that contain different number of nodes, 
        When neuron is approximated by the boxes of given length.

        Parameters:
        -----------
        box_length: float > 0
            the length of box.
        nodes_type: str
            it can be 'all', 'tips', 'branches' and 'tips and branches' 

        Return:
        -------
        n_box_vs_containing: numpy
            an array that at location k says how many boxes contain exatly k nodes of neuron
            (for 0 <= k <=  maximum number of nodes of neuron that a box can contain)

        """
        nodes_in_boxes = self.boxing(box_length=box_length,
                                     return_counts=True,
                                     nodes_type=nodes_type)[1]
        n_box_vs_containing = np.histogram(nodes_in_boxes, 
                                   bins=range(0, nodes_in_boxes.max()+1))[0]
        return n_box_vs_containing

    def set_ratio_red_to_ext(self,c):
        self.ratio_red_to_ext = c

    def set_ext_red_list(self):
        """
        In the extension-reduction perturbation, one of the node will be removed or one node will be added. In the first case, the node can only be
        an end point, but in the second case the new node might be added to any node that has one or zero child.

        dependency:
            self.nodes_list
            tree_util.branch_order
            self.n_soma
            self.ratio_red_to_ext

        ext_red_list:
            first row: end points and order one nodes (for extension)
            second row: end points (for removing)
            third row: end point wich thier parents are order one nodes (for extension)

        Remarks:
            1) The list is zero for the soma nodes.
            2) The value for first and second is binary but the third row is self.ratio_red_to_ext
        """
        (I,) = np.where(tree_util.branch_order[self.n_soma:] == 0)
        I = I + self.n_soma
        self.ext_red_list = np.zeros((3, self.n_node))
        self.ext_red_list[0, I] = 1
        self.ext_red_list[0, np.where(tree_util.branch_order == 1)] = 1
        self.ext_red_list[1, I] = self.ratio_red_to_ext
        J = np.array([])
        for i in I:
            if(len((self.nodes_list[i].parent).children) == 1):
                J = np.append(J, i)
        J = np.array(J, dtype=int)
        self.ext_red_list[2, J] = 1
        self.ext_red_list.astype(int)
        self.ext_red_list[:, 0:self.n_soma] = 0

    def set_pictural_tips(self, branch_order, x_mesh, y_mesh):
        image = np.zeros(int(x_mesh * y_mesh))
        (I,) = np.where(branch_order==0)
        X = self.normlize(self.location[0, I], x_mesh-1)
        Y = self.normlize(self.location[1, I], y_mesh-1)
        L = X + x_mesh*Y
        L = np.array(L)
        L = L.astype(int)
        index, count = np.unique(L, return_counts=True)
        image[list(index)] = list(count)
        image = image/sum(image)
        return image

    def distance_from_root(self):
        """
        Set the distance of each nodes from the root.
        dependency:
            self.location
        """
        distance_from_root = np.sqrt(sum(self.location ** 2))
        return distance_from_root

    def distance_from_parent(self):
        """
        dependency:
            self.location
            self.parent_index
        """
        a = (self.location - self.location[:, self.parent_index.astype(int)]) ** 2
        distance_from_parent = np.sqrt(sum(a))
        return distance_from_parent

    def set_branch_angle_segment(self, important_node, parent_important):
        I = np.array([])
        for i in important_node:
            (J,) = np.where(parent_important == i)
            if(len(J) == 2):
                vec0 = np.expand_dims(self.location[:,important_node[J[0]]] - self.location[:,i], axis = 1)
                vec1 = np.expand_dims(self.location[:,important_node[J[1]]] - self.location[:,i], axis = 1)
                I = np.append(I,self.angle_vec_matrix(vec0,vec1))
        return I

    def branch_angle(self, branch_order):
        """
        An array with size [3, n_nodes] and shows the angles at the branching
        nodes. First row is the angle of two outward segments at the branching
        point Second and third rows are the angle betwen two outward segments
        and previous segment at the branching in arbitrary order (nan at other
        nodes).

        dependency:
            tree_util.branch_order
            self.location
            self.parent_index
            self.n_soma
        """
        (I,) = np.where(branch_order[self.n_soma:] == 2)
        I += self.n_soma
        if(len(I) != 0):
            child_index = np.array(map(lambda i: list(np.where(self.parent_index==i)[0]), I)).T
            vec0 = self.location[:, child_index[0, :]]-self.location[:, I]
            vec1 = self.location[:, child_index[1, :]] - self.location[:, I]
            vec2 = self.location[:, self.parent_index[I].astype(int)] - self.location[:, I]
            branch_angle = self.angle_vec_matrix(vec0, vec1)
            side_angle1 = self.angle_vec_matrix(vec0, vec2)
            side_angle2 = self.angle_vec_matrix(vec2, vec1)
        else:
            branch_angle = np.array([0])
            side_angle1 = np.array([0])
            side_angle2 = np.array([0])

        side_angle = np.append(side_angle1, side_angle2)
        return branch_angle, side_angle

    def global_angle(self):
        """
        dependency:
            sefl.location
            self.parent_index
            self.n_soma
        """
        direction = self.location - self.location[:, self.parent_index]
        global_angle = np.pi - self.angle_vec_matrix(self.location, direction)
        return global_angle

    def local_angle(self, branch_order):
        """
        dependency:
            self.location
            self.n_soma
            self.parent_index
        """
        (I,) = np.where(branch_order[self.n_soma:] == 1)
        I = I + self.n_soma
        child_index = map(lambda i: np.where(self.parent_index == i)[0][0], I)
        dir1 = self.location[:, I] - self.location[:, self.parent_index[I]]
        dir2 = self.location[:, I] - self.location[:, child_index]
        local_angle = self.angle_vec_matrix(dir1, dir2)
        if(len(local_angle) == 0):
            local_angle = np.array([0])
        return local_angle

    def dendrogram_depth(self, parent_index):
        n = len(parent_index)
        depth = np.zeros(n)
        for i in range(1, n):
            depth[i] = depth[parent_index[i]] + 1
        return depth.astype(int)

    def type_at_depth(self,
                      branch_order,
                      parent_index,
                      branch_branch=[],
                      branch_die=[],
                      die_die=[]):
        dead_index = np.where(branch_order == 0)[0]
        continue_index = np.where(branch_order == 1)[0]
        branch_index = np.where(branch_order == 2)[0]
        depth_all = self.dendrogram_depth(parent_index)

        m = depth_all.max()+1
        branch_depth = np.zeros(m)
        dead_depth = np.zeros(m)
        continue_depth = np.zeros(m)
        branch_branch_depth = np.zeros(m)
        branch_die_depth = np.zeros(m)
        die_die_depth = np.zeros(m)

        unique, counts = np.unique(depth_all[dead_index], return_counts=True)
        dead_depth[unique] = counts

        unique, counts = np.unique(depth_all[branch_index], return_counts=True)
        branch_depth[unique] = counts

        unique, counts = np.unique(depth_all[continue_index], return_counts=True)
        continue_depth[unique] = counts

        if len(branch_branch) + len(branch_die) +  len(die_die) != 0:
            unique, counts = np.unique(depth_all[branch_branch], return_counts=True)
            branch_branch_depth[unique] = counts

            unique, counts = np.unique(depth_all[branch_die], return_counts=True)
            branch_die_depth[unique] = counts

            unique, counts = np.unique(depth_all[die_die], return_counts=True)
            die_die_depth[unique] = counts

        return branch_depth, continue_depth, dead_depth, branch_branch_depth, branch_die_depth, die_die_depth

    def set_frustum(self):
        """
        dependency:
            self.distance_from_parent
            self.n_soma
            self.diameter
            self.parent_index
        """
        self.frustum = np.array([0])
        l = self.distance_from_parent[self.n_soma:]
        r = self.diameter[self.n_soma:]
        R = self.diameter[self.parent_index][self.n_soma:]
        f = (np.pi/3.0)*l*(r ** 2 + R ** 2 + r * R)
        self.frustum = np.append(np.zeros(self.n_soma), f)

    def set_curvature(self):
        """
        dependency:
            parent_index
            location
            n_soma
        """
        par = self.parent_index
        papar = par[par]
        papapar = par[par[par]]
        dir1 = self.location[:, par] - self.location
        dir2 = self.location[:, papar] - self.location[:, par]
        dir3 = self.location[:, papapar] - self.location[:, papar]
        cros1 = np.cross(np.transpose(dir1), np.transpose(dir2))
        cros2 = np.cross(np.transpose(dir2), np.transpose(dir3))
        curvature = self.angle_vec_matrix(np.transpose(cros1), np.transpose(cros2))
        return curvature

    def diameter_euclidean(self, distance_from_root, bins=10):
        maximum = max(distance_from_root)
        M = np.zeros(bins)
        for i in range(bins):
            (index_less,) = np.where(distance_from_root < (i+1)*.1*maximum)
            (index_higher,) = np.where(i*.1*maximum < distance_from_root)
            index = np.intersect1d(index_less, index_higher)
            dia = self.diameter[index]
            if(len(dia)==0):
                m = 0
            else:
                m = dia.mean()
            M[i] = m
        return M

    def set_rall_ratio(self):
        """
        dependency:
            self.diameter
            self.child_index
            self.n_soma
            self.n_node
        """
        self.rall_ratio = np.nan*np.ones(self.n_node)
        (I,) = np.where(tree_util.branch_order[self.n_soma:] == 2)
        ch1 = np.power(self.diameter[self.child_index[0,I]],2./3.)
        ch2 = np.power(self.diameter[self.child_index[1,I]],2./3.)
        n = np.power(self.diameter[I],2./3.)
        self.rall_ratio[I] = (ch1+ch2)/n

    def set_values_ite(self):
        """
        set iteratively the following attributes:
            parent_index
            child_index
            location
            diameter
            rall_ratio
            distance_from_root
            distance_from_parent
            slope
            branch_angle
            branch_order
        """
        self.parent_index = np.zeros(self.n_soma)
        self.child_index = np.nan * np.ones([2,self.n_soma])
        for n in self.nodes_list[1:]:
            self.location = np.append(self.location, n.xyz.reshape([3,1]), axis = 1)
            self.diameter = np.append(self.diameter, n.r)
        for n in self.nodes_list[1:]:
            #self.frustum = np.append(self.frustum,  self.calculate_frustum(n))
            #self.rall_ratio = np.append(self.rall_ratio, self.calculate_rall(n))
            self.distance_from_root = np.append(self.distance_from_root, self.calculate_distance_from_root(n))
            self.distance_from_parent = np.append(self.distance_from_parent, self.calculate_distance_from_parent(n))
            #self.slope = np.append(self.slope, self.calculate_slope(n))
            ang, ang1, ang2 = self.calculate_branch_angle(n)
            an = np.zeros([3,1])
            an[0,0] = ang
            an[1,0] = ang1
            an[2,0] = ang2

            if(self.branch_angle.shape[1] == 0):
                self.branch_angle = an
            else:
                self.branch_angle = np.append(self.branch_angle, an, axis = 1)
            glob_ang, local_ang = self.calculate_node_angles(n)
            self.global_angle = np.append(self.global_angle, glob_ang)
            self.local_angle = np.append(self.local_angle, local_ang)
            #self.neural_distance_from_soma = np.append(self.neural_distance_from_soma, self.calculate_neural_distance_from_soma(n))
        for n in self.nodes_list[self.n_soma:]:
            self.parent_index = np.append(self.parent_index, self.get_index_for_no_soma_node(n.parent))
            if(tree_util.branch_order[self.get_index_for_no_soma_node(n)]==2):
                a = np.array([self.get_index_for_no_soma_node(n.children[0]),self.get_index_for_no_soma_node(n.children[1])]).reshape(2,1)
                self.child_index = np.append(self.child_index, a, axis = 1)
            if(tree_util.branch_order[self.get_index_for_no_soma_node(n)]==1):
                a = np.array([self.get_index_for_no_soma_node(n.children[0]),np.nan]).reshape(2,1)
                self.child_index = np.append(self.child_index, a, axis = 1)
            if(tree_util.branch_order[self.get_index_for_no_soma_node(n)]==0):
                a = np.array([np.nan,np.nan]).reshape(2,1)
                self.child_index = np.append(self.child_index, a, axis = 1)

    def set_parent(self):
        self.parent_index = np.zeros(self.n_soma)
        self.child_index = np.zeros([2,self.n_node])
        for n in self.nodes_list[self.n_soma:]:
            par = self.get_index_for_no_soma_node(n.parent)
            node = self.get_index_for_no_soma_node(n)
            self.parent_index = np.append(self.parent_index, par)
            if self.child_index[0,par] != 0:
                self.child_index[1,par] = node
            else:
                self.child_index[0,par] = node
        self.child_index[self.child_index == 0] = np.nan
        self.child_index[:,0:self.n_soma] = np.nan
        self.parent_index = np.array(self.parent_index, dtype=int)
        #self.parent_index.astype(int)

    def set_loc_diam(self):
        self.location = np.zeros([3,self.n_node])
        self.diameter = np.zeros(self.n_node)
        for n in range(self.n_node):
            self.location[:,n] = self.nodes_list[n].xyz
            self.diameter[n] = self.nodes_list[n].r

    def set_connection(self):
        """
        Set the full connection matrix for neuron. connection is an array with
        size [n_node, n_node]. The element (i,j) is not np.nan if node i is a
        decendent of node j. The value at this array is the distance of j to
        its parent.

        dependency:
            self.nodes_list
            self.n_soma
            self.n_node
            self.parent_index
            self.distance_from_parent
        """
        connection = np.zeros([self.n_node, self.n_node])
        connection[np.arange(self.n_node), self.parent_index.astype(int)] = 1
        connection[0, 0] = 0
        connection = inv(np.eye(self.n_node) - connection)
        connection[connection != 1] = np.nan
        for i in range(self.n_node):
            (J,) = np.where(~np.isnan(connection[:, i]))
            connection[J, i] = self.features['distance from parent'][i]
        som = np.eye(self.n_soma)
        som[som != 1] = np.nan
        connection[np.ix_(np.arange(self.n_soma), np.arange(self.n_soma))] = som
        connection[:, 0] = 1.
        self.connection = connection

    def set_connection_old(self):
        """

        dependency:
            self.nodes_list
            self.n_soma
            self.parent_index
            self.distance_from_parent
        """
        self.parent_index = np.array(self.parent_index, dtype = int)
        L = self.n_node - self.n_soma
        C = csr_matrix((np.ones(L),(range(self.n_soma,self.n_node), self.parent_index[self.n_soma:])), shape = (self.n_node,self.n_node))
        self.connection = np.zeros([self.n_node,self.n_node]) # the connectivity matrix
        new = 0
        i = 0
        old = C.sum()
        while(new != old):
            self.connection = C.dot(csr_matrix(self.connection)) + C
            old = new
            new = self.connection.sum()
        self.connection = self.connection.toarray()
        self.connection[range(1,self.n_node),range(1,self.n_node)] = 1
        self.connection[:,:self.n_soma] = 0

        # fill the matrix with the distance
        for i in range(self.n_node):
            self.connection[self.connection[:,i] != 0,i] = self.distance_from_parent[i]
        self.connection[self.connection == 0] = np.nan

    def set_sholl(self):
        self.sholl_r = np.array([])
        for n in self.nodes_list:
            dis = LA.norm(self.xyz(n) - self.root.xyz,2)
            self.sholl_r = np.append(self.sholl_r, dis)

        self.sholl_r = np.sort(np.array(self.sholl_r))
        self.sholl_n = np.zeros(self.sholl_r.shape)
        for n in self.nodes_list:
            if(n.parent != None):
                par = n.parent
                dis_par = LA.norm(self.xyz(par) - self.root.xyz,2)
                dis_n = LA.norm(self.xyz(par) - self.root.xyz,2)
                M = max(dis_par, dis_n)
                m = min(dis_par, dis_n)
                I = np.logical_and(self.sholl_r>=m, self.sholl_r<=M)
                self.sholl_n[I] = self.sholl_n[I] + 1

    def set_location(self):
        self.location = np.zeros([3, len(self.nodes_list)])
        for i in range(len(self.nodes_list)):
            self.location[:, i] = self.nodes_list[i].xyz

    def set_type(self):
        self.nodes_type = np.zeros(len(self.nodes_list))
        for i in range(len(self.nodes_list)):
            self.nodes_type[i] = self.nodes_list[i].node_type

    def xyz(self, node):
        return self.location[:,self.get_index_for_no_soma_node(node)]

    def _r(self, node):
        return self.diameter[self.get_index_for_no_soma_node(node)]

    def parent_index_for_node_subset(self, subset):
        """
        inputs
        ------
            index of subset of the nodes without root node
        output
        ------
            Index of grand parent inside of the subset for each member of subset
        """
        if((subset == 0).sum() == 0):
            subset = np.append(0, subset)
        n = subset.shape[0]
        A = self.connection[np.ix_(subset, subset)]
        A[np.isnan(A)] = 0
        A[A != 0] = 1.
        B = np.eye(subset.shape[0]) - inv(A)
        return subset[np.where(B==1)[1]]

    def distance(self, index1, index2):
        """
        Neural distance between two nodes in the neuron.

        inputs
        ------
            index1, index2 : the indecies of the nodes.
        output
        ------
            the neural distance between the node.
        """
        return min(self.distance_two_node_up_down(index1,index2),self.distance_two_node_up_down(index2,index1))

    def distance_two_node_up_down(self, Upindex, Downindex):
        (up,) = np.where(~np.isnan(self.connection[Downindex,:]))
        (down,) = np.where(~np.isnan(self.connection[:,Upindex]))
        I = np.intersect1d(up,down)
        if(I.shape[0] != 0):
            return sum(self.distance_from_parent[I]) - self.distance_from_parent[Upindex]
        else:
            return np.inf

    def calculate_overall_matrix(self, node):
        j = self.get_index_for_no_soma_node(node)
        k = self.get_index_for_no_soma_node(node.parent)
        (J,)  = np.where(~ np.isnan(self.connection[:,j]))
        dis = LA.norm(self.location[:,k] - self.location[:,j],2)
        self.connection[J,j] = dis

    def calculate_branch_order(self,node):
        """
        terminal = 0, passig (non of them) = 1, branch = 2
        """
        return len(node.children)

    def calculate_frustum(self,node):
        """
        the Volume of the frustum ( the node with its parent) at each location. (nan for the nodes of soma)
        """
        r = self._r(node)
        r_par = self._r(node.parent)
        dis = LA.norm(self.xyz(node) - self.xyz(node.parent) ,2)
        f = dis*(np.pi/3.0)*(r*r + r*r_par + r_par*r_par)
        return f

    def calculate_rall(self,node):
        if(len(node.children) == 2):
            n1, n2 = node.children
            r1 = self._r(n1)
            r2 = self._r(n2)
            r = self._r(node)
            rall = (np.power(r1,2.0/3.0)+(np.power(r2,2.0/3.0)))/np.power(r,2.0/3.0)
        else:
            rall = np.nan
        return rall

    def calculate_distance_from_root(self,node):
        return LA.norm(self.xyz(node) - self.root.xyz,2)

    def calculate_distance_from_parent(self,node):
        return LA.norm(self.xyz(node) - self.xyz(node.parent),2)

    def calculate_slope(self,node):
        # the ratio of: delta(pos)/delta(radius)
        dis = LA.norm(self.xyz(node) - self.xyz(node.parent),2)
        rad = node.r - node.parent.r
        if(dis == 0):
            val = rad
        else:
            val = rad/dis
        return val

    def calculate_branch_angle(self, node):
        """
        Calculate the mean of the angle betwen two outward segments and
        previous segment at the branching (nan at other nodes).
        """
        if(len(node.children) == 2):
            n1, n2 = node.children
            nodexyz = self.xyz(node)
            node_parxyz = self.xyz(node.parent)
            node_chixyz1 = self.xyz(n1)
            node_chixyz2 = self.xyz(n2)
            vec = node_parxyz - nodexyz
            vec1 = node_chixyz1 - nodexyz
            vec2 = node_chixyz2 - nodexyz
            ang = self.angle_vec(vec1, vec2)
            ang1 = self.angle_vec(vec1, vec)
            ang2 = self.angle_vec(vec2, vec)
        else:
            ang = np.nan
            ang1 = np.nan
            ang2 = np.nan
        return ang, ang1, ang2

    def calculate_node_angles(self,node):
        par = node.parent
        nodexyz = self.xyz(node)
        node_parxyz = self.xyz(node.parent)
        vec1 = node_parxyz - nodexyz
        vec2 = self.root.xyz - nodexyz
        glob_ang = self.angle_vec(vec1,vec2)
        if(node.children != None):
            if(len(node.children) ==1):
                [child] = node.children
                vec3 = self.xyz(child) - nodexyz
                local_ang = self.angle_vec(vec1,vec3)
            else:
                local_ang = np.nan
        else:
            local_ang = np.nan
        return glob_ang, local_ang

    def adjust_connection_with_parent(self):
        """
        Set the connection matrix by distance from parent. Notice that the
        positions of nan in the connection matrix should be correct. This
        function set the right value to the numerical values of the connection
        matrix.
        """
        self.connection[~np.isnan(self.connection)] = 1.
        self.connection = self.connection*self.distance_from_parent

    def angle_vec_matrix(self,matrix1,matrix2):
        """
        Takes two matrix 3*n of matrix1 and matrix2 and gives back
        the angles for each corresponding n vectors.
        Note: if the norm of one of the vectors is zeros the angle is np.pi
        """
        ang = np.zeros(matrix1.shape[1])
        norm1 = LA.norm(matrix1, axis = 0)
        norm2 = LA.norm(matrix2, axis = 0)
        domin = norm1*norm2
        (J,) = np.where(domin != 0)
        ang[J] = np.arccos(np.maximum(np.minimum(sum(matrix1[:,J]*matrix2[:,J])/domin[J],1),-1))
        return ang

    def angle_vec(self, vec1, vec2):
        val = sum(vec1*vec2)/(LA.norm(vec1, 2)*LA.norm(vec2, 2))
        if(LA.norm(vec1, 2) == 0 or LA.norm(vec2, 2) == 0):
            val = -1
        return math.acos(max(min(val, 1), -1))

    def choose_random_node_index(self):
            n = np.floor((self.n_node-self.n_soma)*np.random.random_sample()).astype(int)
            return n + self.n_soma

    def possible_ext_red_whole(self):
        """
        Thos function gives back the probabiliy of the chossing one of the node add_node
        extend it.
        """
        return self.ext_red_list[0:2,:].sum()+1 # 1 added because the root may extend

    def possible_ext_red_end_point(self):
        """
        Those function gives back the probabiliy of the chossing one of the node add_node
        extend it.
        """
        return self.ext_red_list[1:3,:].sum()

    def get_index_for_no_soma_node(self,node):
        return self.nodes_list.index(node)

    def _list_for_local_update(self,node):
        """
        Return the index of node, its parent and any children it may have.
        The node should be a no_soma node
        """
        update_list = np.array([]) # index of all nodes for update
        update_list = np.append(update_list, self.get_index_for_no_soma_node(node))
        if(node.parent.node_type != 1):
            update_list = np.append(update_list, self.get_index_for_no_soma_node(node.parent)) # if the node doesnt have a parent in no_soma list, i.e. its parent is a soma, get_index would return nothing
        if(node.children != None):
            for n in node.children:
                update_list = np.append(update_list, self.get_index_for_no_soma_node(n))
        return update_list.astype(int)

    def _update_attribute(self, update_list):
        for ind in update_list:
            #self.frustum[ind] = self.calculate_frustum(self.nodes_list[ind])
            #self.rall_ratio[ind] = self.calculate_rall(self.nodes_list[ind])
            self.distance_from_root[ind] =  self.calculate_distance_from_root(self.nodes_list[ind])
            self.distance_from_parent[ind] = self.calculate_distance_from_parent(self.nodes_list[ind])
            #self.slope[ind] = self.calculate_slope(self.nodes_list[ind])
            tree_util.branch_order[ind] = self.calculate_branch_order(self.nodes_list[ind])
            ang, ang1, ang2 = self.calculate_branch_angle(self.nodes_list[ind])
            self.branch_angle[0, ind] = ang
            self.branch_angle[1, ind] = ang1
            self.branch_angle[2, ind] = ang2
            ang1, ang2 = self.calculate_node_angles(self.nodes_list[ind])
            self.global_angle[ind] = ang1
            self.local_angle[ind] = ang2
            self.calculate_overall_matrix(self.nodes_list[ind])
        #self.sholl_r = np.array([]) # the position of the jumps for sholl analysis
        #self.sholl_n = np.array([]) # the value at the jumping (the same size as self.sholl_x)

    def change_location(self,index,displace):
        """
        Change the location of one of the node in the neuron updates the attribute accordingly.
        Parameters:
        ___________
        index: the index of node in no_soma_list to change its diameter

        displace: the location of new node is the xyz of the current locatin + displace
        """
        # First change the location of the node by displace
        node = self.nodes_list[index]
        self.location[:,index] += displace
        self._update_attribute(self._list_for_local_update(node))
        self.set_features()

    def change_location_toward_end_nodes(self,index,displace):
        (I,) = np.where(~np.isnan(self.connection[:,index]))
        self.location[0,I] += displace[0]
        self.location[1,I] += displace[1]
        self.location[2,I] += displace[2]
        #self.set_distance_from_root()
        #self.set_distance_from_parent()
        self.connection[np.ix_(I,[index])] = self.distance_from_parent[index]
        self.set_branch_angle()
        self.set_global_angle()
        self.set_local_angle()
        self.set_features()

    def change_location_important(self, index, displace):
        (branch_index,)  = np.where(tree_util.branch_order[self.n_soma:]==2)
        (end_nodes,)  = np.where(tree_util.branch_order[self.n_soma:]==0)
        branch_index += self.n_soma
        end_nodes += self.n_soma
        I = np.append(branch_index, end_nodes)
        parents = self.parent_index_for_node_subset(I)
        (ind,) = np.where(I == index)
        origin = deepcopy(self.location[:,index])
        # correct the segment to the parent
        par = parents[ind][0]
        (up,) = np.where(~np.isnan(self.connection[index,:]))
        (down,) = np.where(~np.isnan(self.connection[:,par]))
        J = np.intersect1d(up,down)
        A = self.location[:,J]
        loc = self.location[:,par]
        A[0,:] = A[0,:] - loc[0]
        A[1,:] = A[1,:] - loc[1]
        A[2,:] = A[2,:] - loc[2]
        r1 = origin - loc
        r2 = r1 + displace
        M = self.scalar_rotation_matrix_to_map_two_vector(r1, r2)
        A = np.dot(M,A)
        A[0,:] = A[0,:] + loc[0]
        A[1,:] = A[1,:] + loc[1]
        A[2,:] = A[2,:] + loc[2]
        self.location[:,J] = A
        changed_ind = J
        # correct the children
        (ch,) = np.where(parents == index)
        for i in I[ch]:
            (up,) = np.where(~np.isnan(self.connection[i,:]))
            (down,) = np.where(~np.isnan(self.connection[:,index]))
            J = np.intersect1d(up,down)
            A = self.location[:,J]
            loc = self.location[:,i]
            A[0,:] = A[0,:] - loc[0]
            A[1,:] = A[1,:] - loc[1]
            A[2,:] = A[2,:] - loc[2]
            r1 = origin - loc
            r2 = r1 + displace
            M = self.scalar_rotation_matrix_to_map_two_vector( r1, r2)
            A = np.dot(M,A)
            A[0,:] = A[0,:] + loc[0]
            A[1,:] = A[1,:] + loc[1]
            A[2,:] = A[2,:] + loc[2]
            self.location[:,J] = A
            changed_ind = np.append(changed_ind, J)
        self.location[:,index] = origin + displace
        #self.set_distance_from_root()
        #self.set_distance_from_parent()
        for i in changed_ind:
            (J,) = np.where(~np.isnan(self.connection[:,i]))
            self.connection[J,i] = self.distance_from_parent[i]
        self.set_branch_angle()
        self.set_global_angle()
        self.set_local_angle()
        self.set_features()

    def scalar_rotation_matrix_to_map_two_vector(self, v1, v2):
        r1 = LA.norm(v1,2)
        norm1 = v1/r1
        r2 = LA.norm(v2,2)
        normal2 = v2/r2
        a = sum(normal2*norm1)
        theta = -np.arccos(a)
        normal2 = normal2 - a*norm1
        norm2 = normal2/LA.norm(normal2,2)
        cross = np.cross(norm1, norm2)
        B = np.zeros([3,3])
        B[:,0] = norm1
        B[:,1] = norm2
        B[:,2] = cross
        A = np.eye(3)
        A[0,0] = np.cos(theta)
        A[1,0] = - np.sin(theta)
        A[0,1] = np.sin(theta)
        A[1,1] = np.cos(theta)
        return (r2/r1) * np.dot(np.dot(B,A),inv(B))

    def change_diameter(self, index, ratio):
        """
        Change the diameter of one node in the neuron updates the attribute accordingly.
        Parameters:
        ___________
        index: the index of node in no_soma_list to change its diameter

        ratio: the radius of new node is the radius of current node times ratio
        """
        node = self.nodes_list[index]
        node.r = ratio*node.r
        r = node.r
        self.diameter[index] = r
        self._update_attribute(self._list_for_local_update(node))
        self.set_features()

    def change_diameter_toward(self, index, ratio):
        """
        Change the diameter of one node in the neuron updates the attribute accordingly.
        Parameters:
        ___________
        index: the index of node in no_soma_list to change its diameter

        ratio: the radius of new node is the radius of current node times ratio
        """
        (index_toward_soma,) = np.where(~np.isnan(self.connection[index, self.n_soma:]))
        index_toward_soma = index_toward_soma + self.n_soma
        self.diameter[index_toward_soma] = ratio * self.diameter[index_toward_soma]
        # self._update_attribute(self._list_for_local_update(node))
        # self.set_features()

    def rescale_toward_end(self,node, rescale):
        """
        Rescale the part of neuron form the node toward the end nodes.
        input
        -----
        node : `Node` class
            the node of the neuron which the location of other nodes in the neuron have it as thier decendent would be changed.
            rescale : positive float
            The value to rescale the part of the neuron.
        """
        index = self.get_index_for_no_soma_node(node)
        (I,) = np.where(~np.isnan(self.connection[:,index]))
        A = self.location[:,I]
        loc = self.xyz(node)
        A[0,:] = A[0,:] - loc[0]
        A[1,:] = A[1,:] - loc[1]
        A[2,:] = A[2,:] - loc[2]
        A = rescale*A
        A[0,:] = A[0,:] + loc[0]
        A[1,:] = A[1,:] + loc[1]
        A[2,:] = A[2,:] + loc[2]
        self.location[:,I] = A
        #self.set_distance_from_root()
        #self.set_distance_from_parent()
        I = I.tolist()
        I.remove(index)
        I = np.array(I,dtype = int)
        self.connection[:,I] *= rescale
        self.set_branch_angle()
        self.set_global_angle()
        self.set_local_angle()
        self.set_features()

    def rotate(self, node_index, matrix):
        """
        Rotate the neuron around the parent of the node with the given matrix.
        The attribute to update:
            location
            distance_from_root
            branch_angle
            angle_global
            local_angle
        """
        parent_index = self.parent_index[node_index]
        index_afterward_nodes = self.connecting_after_node(parent_index)
        A = self.location[:, index_afterward_nodes]
        loc = self.location[:, parent_index]
        A[0, :] = A[0, :] - loc[0]
        A[1, :] = A[1, :] - loc[1]
        A[2, :] = A[2, :] - loc[2]
        A = np.dot(matrix, A)
        A[0, :] = A[0, :] + loc[0]
        A[1, :] = A[1, :] + loc[1]
        A[2, :] = A[2, :] + loc[2]
        self.location[:, index_afterward_nodes] = A
        self.set_features()

    def remove_node(self, index):
        """
        Removes a non-soma node from the neuron and updates the features

        Parameters
        ----------
        Node : the index of the node in the no_soma_list
            the node should be one of the end-points, otherwise gives an error
        """
        self.n_node -= 1
        node = self.nodes_list[index]
        parent_index = self.get_index_for_no_soma_node(node.parent)
        # details of the removed node for return
        p = node.parent
        node.parent.remove_child(node)
        l = self.location[:, index] - self.location[:, parent_index]
        r = self.diameter[index]/self.diameter[parent_index]
        self.location = np.delete(self.location, index, axis=1)
        self.nodes_type = np.delete(self.nodes_type, index)
        self.nodes_list.remove(node)
        tree_util.branch_order = np.delete(tree_util.branch_order, index)
        new_parent_index = self.get_index_for_no_soma_node(p)
        tree_util.branch_order[new_parent_index] -= 1

        self.diameter = np.delete(self.diameter,index)
        #self.frustum = np.delete(self.frustum,index)
        #self.rall_ratio = np.delete(self.rall_ratio,index)
        self.distance_from_root = np.delete(self.distance_from_root,index)
        self.distance_from_parent = np.delete(self.distance_from_parent,index)
        #self.slope = np.delete(self.slope,index)
        self.branch_angle = np.delete(self.branch_angle,index, axis = 1)
        self.global_angle = np.delete(self.global_angle,index)
        self.local_angle = np.delete(self.local_angle,index)
        # self.connection = np.delete(self.connection,index, axis = 0)
        # self.connection = np.delete(self.connection,index, axis = 1)

        self.parent_index = np.delete(self.parent_index,index)
        I = np.where(self.parent_index > index)
        self.parent_index[I] -= 1
        self.child_index = np.delete(self.child_index,index,axis = 1)
        I , J = np.where(self.child_index  > index)
        self.child_index[I,J] -= 1
        if p.return_type_name() is not 'soma':
            if len(p.children) == 1:
                self.branch_angle[0,new_parent_index] = np.nan
                self.branch_angle[1,new_parent_index] = np.nan
                self.branch_angle[2,new_parent_index] = np.nan
                gol, loc = self.calculate_node_angles(self.nodes_list[new_parent_index])
                self.child_index[:,new_parent_index] = np.array([self.get_index_for_no_soma_node(p.children[0]), np.nan])
                self.local_angle[new_parent_index] = loc
            if len(p.children) == 0:
                self.local_angle[new_parent_index] = np.nan
                self.child_index[:,new_parent_index] = np.array([np.nan, np.nan])

        #self.sholl_r = None # the position of the jumps for sholl analysis
        #self.sholl_n = None # the value at the jumping (the same size as self.sholl_x)
        self.set_ext_red_list()
        self.set_features()
        return p, l, r

    def extend_node(self, parent, location, ratio):
        """
        Extend the neuron by adding one end point and updates the attribute
        for the new neuron.

        Parameters:
        -----------
        Parent: Node
            the node that the extended node attached to
        location: numpy array
            the xyz of new node is the sum of location and xyz of parent
        ratio: floa
            radius of new node is the radius of parent times ratio
        """
        self.n_node += 1
        if parent is 'soma':
            parent = self.root
        n = Node()
        in_par = self.get_index_for_no_soma_node(parent)
        n.node_type = 2
        R = ratio * self.diameter[in_par]
        n.parent = parent
        parent.add_child(n)
        self.nodes_type = np.append(self.nodes_type, 2)
        self.location = np.append(self.location,
                                  (self.location[:, in_par]+location).reshape([3, 1]), axis=1)
        self.diameter = np.append(self.diameter, R)
        self.nodes_list.append(n)
        #self.frustum = np.append(self.frustum,np.nan)
        tree_util.branch_order = np.append(tree_util.branch_order, 0)
        tree_util.branch_order[self.get_index_for_no_soma_node(parent)] += 1
        #self.rall_ratio = np.append(self.rall_ratio ,np.nan)
        self.distance_from_root = np.append(self.distance_from_root, np.nan)
        self.distance_from_parent = np.append(self.distance_from_parent, np.nan)
        #self.slope = np.append(self.slope ,np.nan)
        if(self.branch_angle.shape[1] == 0):
            self.branch_angle = np.nan*np.ones([3, 1])
        else:
            self.branch_angle = np.append(self.branch_angle, np.nan*np.ones([3, 1]), axis=1)
        self.global_angle = np.append(self.global_angle, np.nan)
        self.local_angle = np.append(self.local_angle, np.nan)

        #l = self.connection.shape[0]
        #I = np.nan*np.zeros([1, l])
        #(J,) = np.where(~np.isnan(self.connection[self.get_index_for_no_soma_node(parent), :]))
        #I[0, J] = self.connection[self.get_index_for_no_soma_node(parent), J]
        #self.connection = np.append(self.connection, I, axis=0)
        #self.connection = np.append(self.connection, np.nan*np.zeros([l+1, 1]), axis=1)
        #self.connection[l, l] = LA.norm(location, 2)

        self.parent_index = np.append(self.parent_index, self.get_index_for_no_soma_node(parent))
        self.child_index = np.append(self.child_index, np.array([np.nan, np.nan]).reshape(2,1), axis = 1)
        if parent.return_type_name() is not 'soma':
            if(len(parent.children) == 1):
                self.child_index[:, self.get_index_for_no_soma_node(parent)] = np.array([self.get_index_for_no_soma_node(n), np.nan])
            if(len(parent.children) == 2):
                self.child_index[1, self.get_index_for_no_soma_node(parent)] = self.get_index_for_no_soma_node(n)
        update_list = self._list_for_local_update(n)
        self._update_attribute(update_list)

        self.set_ext_red_list()
        self.set_features()
        return self.get_index_for_no_soma_node(n)

    def add_extra_node(self, node, length, radius):
        index = self.connecting_after_node(node)

    def remove_extra_node(self, node):
        print 1

    def slide(self, detached_node, attached_node):
        """

        """
        index_afterward_nodes = self.connecting_after_node(detached_node)
        parent_detached = self.parent_index[detached_node]
        self.parent_index[detached_node] = attached_node
        for i in range(3):
            self.location[i, index_afterward_nodes] += \
                self.location[i, attached_node] - self.location[i, parent_detached]
        self.set_features()

    def horizental_stretch(self, node_index, parent_node, scale):
        (up,) = np.where(~np.isnan(self.connection[node_index,:]))
        (down,) = np.where(~np.isnan(self.connection[:,parent_node]))
        I = np.intersect1d(up,down)
        A = self.location[:,I]
        loc = self.location[:,parent_node]
        A[0,:] = A[0,:] - loc[0]
        A[1,:] = A[1,:] - loc[1]
        A[2,:] = A[2,:] - loc[2]
        r = self.location[:,node_index] - loc
        r = r/LA.norm(r,2)
        A = scale*A +(1-scale)*(np.dot(np.expand_dims(r,axis = 1),np.expand_dims(np.dot(r,A),axis = 0)))
        A[0,:] = A[0,:] + loc[0]
        A[1,:] = A[1,:] + loc[1]
        A[2,:] = A[2,:] + loc[2]
        self.location[:,I] = A
        self.set_distance_from_root()
        self.set_distance_from_parent()
        for i in I:
            (J,) = np.where(~np.isnan(self.connection[:,i]))
            self.connection[J,i] = self.distance_from_parent[i]
        self.set_branch_angle()
        self.set_global_angle()
        self.set_local_angle()
        self.set_features()

    def vertical_stretch(self, node_index, parent_node, scale):
        (up,) = np.where(~np.isnan(self.connection[node_index,:]))
        (down,) = np.where(~np.isnan(self.connection[:,parent_node]))
        I = np.intersect1d(up,down)
        A = self.location[:,I]
        loc = self.location[:,parent_node]
        A[0,:] = A[0,:] - loc[0]
        A[1,:] = A[1,:] - loc[1]
        A[2,:] = A[2,:] - loc[2]
        r = self.location[:,node_index] - loc
        new_loc = -(1-scale)*(r)
        r = r/LA.norm(r,2)
        A = A -(1-scale)*(np.dot(np.expand_dims(r,axis = 1),np.expand_dims(np.dot(r,A),axis = 0)))
        A[0,:] = A[0,:] + loc[0]
        A[1,:] = A[1,:] + loc[1]
        A[2,:] = A[2,:] + loc[2]
        self.location[:,I] = A
        (T,) = np.where(~np.isnan(self.connection[:, node_index]))
        T = list(T)
        T.remove(node_index)
        A = self.location[:,T]
        A[0,:] += new_loc[0]
        A[1,:] += new_loc[1]
        A[2,:] += new_loc[2]
        self.location[:,T] = A
        self.set_distance_from_root()
        self.set_distance_from_parent()
        T = np.array(T, dtype=int)
        for i in np.append(T, I):
            (J,) = np.where(~np.isnan(self.connection[:, i]))
            self.connection[J, i] = self.distance_from_parent[i]
        self.set_branch_angle()
        self.set_global_angle()
        self.set_local_angle()
        self.set_features()

    def connecting_after_node(self, node_index):
        """
        Return the index of nodes after the given node toward the end nodes.

        Parameters
        ----------
        node: Node

        Returns
        -------
        index: numpy array
            the index of conneting nodes toward the end (including the node).
        """
        #(index, ) = np.where(~np.isnan(self.connection[:, self.get_index_for_no_soma_node(node)]))
        up = np.arange(self.n_node)
        if(node_index < self.n_soma):
            index_afterward_nodes = up
        else:
            index = np.zeros(self.n_node)
            while(up.sum() != 0):
                index[np.where(up == node_index)] = 1
                up = self.parent_index[up]
            index_afterward_nodes = np.where(index)[0]
        return index_afterward_nodes

    def get_root(self):

        """
        Obtain the root Node

        Returns
        -------
        root : :class:`Node`
        """
        return self.__root

    def is_root(self, node):

        """
        Check whether a Node is the root Node

        Parameters
        -----------
        node : :class:`Node`
            Node to be check if root

        Returns
        --------
        is_root : boolean
            True is the queried Node is the root, False otherwise
        """
        if node.parent is None:
            return True
        else:
            return False

    def is_leaf(self, node):

        """
        Check whether a Node is a leaf Node, i.e., a Node without children

        Parameters
        -----------
        node : :class:`Node`
            Node to be check if leaf Node

        Returns
        --------
        is_leaf : boolean
            True is the queried Node is a leaf, False otherwise
        """
        if len(node.children) == 0:
            return True
        else:
            return False

    def is_branch(self, node):

        """
        Check whether a Node is a branch Node, i.e., a Node with two children

        Parameters
        -----------
        node : :class:`Node`
            Node to be check if branch Node

        Returns
        --------
        is_leaf : boolean
            True is the queried Node is a branch, False otherwise
        """

        if len(node.children) == 2:
            return True
        else:
            return False

    def find_root(self, node):
        if node.parent is not None:
            node = self.find_root(node.parent)
        return node

    def add_node_with_parent(self, node, parent):

        """
        Add a Node to the tree under a specific parent Node

        Parameters
        -----------
        node : :class:`Node`
            Node to be added
        parent : :class:`Node`
            parent Node of the newly added Node
        """
        node.parent = parent
        if parent is not None:
            parent.add_child(node)
        self.add_node(node)

    def add_node(self,node):
        self.nodes_list.append(node)

    def set_nodes_list(self):
        self.nodes_list = []
        self.root = Node()
        self.root.xyz = self.location[:, 0]
        self.root.r = self.diameter[0]
        self.root.node_type = 1
        self.nodes_list.append(self.root)
        for i in range(1, self.n_node):
            node = Node()
            node.xyz = self.location[:, i] - self.location[:, 0]
            node.r = self.diameter[i]
            node.node_type = self.nodes_type[i]
            self.nodes_list.append(node)

        self.child_index = np.nan*np.zeros([2, self.n_node])

        for i in range(1, self.n_node):
            i = int(i)
            p = int(self.parent_index[i])
            if(np.isnan(self.child_index[0, p])):
                self.child_index[0, p] = i
            else:
                self.child_index[1, p] = i

            self.nodes_list[i].parent = self.nodes_list[p]
            self.nodes_list[p].add_child(self.nodes_list[i])

    def read_swc_matrix(self, swc_matrix):
        """
        Read matrix of swc format.
        Root (the node with parent of -1) should be at the first row.

        Parameters
        ----------
        input_file: str
            the address of *.swc file

        Returns
        -------
        The function assigns following attributes:
            n_soma
            n_node
            nodes_list
            location
            nodes_type
            diameter
            parent_index
            child_index
        """
        self.nodes_type = np.squeeze(swc_matrix[:, 1])
        self.nodes_type = np.array(self.nodes_type, dtype=int)
        self.n_node = swc_matrix.shape[0]
        self.location = swc_matrix[:, 2:5].T
        for i in range(3):
            self.location[i, :] = self.location[i, :] - self.location[i, 0]
        (I,) = np.where(self.nodes_type == 1)
        self.n_soma = len(I)
        if(self.n_soma == 0): # root
            self.n_soma = 1
        self.diameter = np.squeeze(swc_matrix[:, 5])
        self.parent_index = swc_matrix[:, 6] - 1
        self.parent_index[0] = 0
        self.parent_index = np.squeeze(self.parent_index)
        self.parent_index = np.array(self.parent_index, dtype=int)
        #self.parent_index = self.parent_index.astype(int)

    def get_swc(self):
        swc = np.zeros([self.n_node,7])
        remain = [self.root]
        index = np.array([-1])
        for i in range(self.n_node):
            n = remain[0]
            swc[i,0] = i+1
            swc[i,1] = n.set_type_from_name()
            ind = self.get_index_for_no_soma_node(n)
            if(ind > self.n_soma):
                swc[i,2] = self.location[0,ind]
                swc[i,3] = self.location[1,ind]
                swc[i,4] = self.location[2,ind]
                swc[i,5] = self.diameter[ind]
                swc[i,6] = index[0]
            else:
                swc[i,2] = n.xyz[0]
                swc[i,3] = n.xyz[1]
                swc[i,4] = n.xyz[2]
                swc[i,5] = n.r
                swc[i,6] = 1
            for m in n.children:
                remain.append(m)
                index = np.append(index,i+1)
            remain = remain[1:]
            index = index[1:]
        swc[0,6] = -1
        return swc

    def get_random_branching_or_end_node(self):
        (b,) = np.where(tree_util.branch_order[self.n_soma:] == 2)
        (e,) = np.where(tree_util.branch_order[self.n_soma:] == 0)
        I = np.append(b, e)
        if(len(I) == 0):
            n = Node()
            n.node_type = 'empty'
        else:
            I += self.n_soma
            i = int(np.floor(len(I)*np.random.rand()))
            n = self.nodes_list[I[i]]
        return n

    def get_random_no_soma_node(self):
        l = self.n_node - self.n_soma
        return (np.floor(l*np.random.rand()) + self.n_soma).astype(int)

    def get_random_branching_node(self, branch_order):
        """
        Return one of the branching point in the neuron.
        dependency:
            tree_util.branch_order
            self.nodes_list
            self.n_soma
        """
        (I,) = np.where(branch_order[self.n_soma:] == 2)
        if(len(I) == 0):
            n = -1
        else:
            I += self.n_soma
            i = np.floor(len(I)*np.random.rand())
            n = I[int(i)]
        return n

    def get_random_order_one_node_not_in_certain_index(self, branch_order, index):
        """
        Return one of the order one point in the neuron.
        dependency:
            tree_util.branch_order
            self.nodes_list
            self.n_soma
        """
        (I,) = np.where(tree_util.branch_order == 1)
        I = I[I >= self.n_soma]
        I = np.setdiff1d(I, index)
        if(len(I) == 0):
            n = -1
        else:
            i = np.floor(len(I)*np.random.rand())
            n = I[int(i)]
        return n

    def get_random_non_branch_node_not_in_certain_index(self, branch_order, index):
        """
        Return one of the order one point in the neuron.
        dependency:
            tree_util.branch_order
            self.nodes_list
            self.n_soma
        """
        (I,) = np.where(branch_order[self.n_soma:] != 2)
        I += self.n_soma
        I = np.setdiff1d(I, index)
        if(len(I) == 0):
            n = -1
        else:
            i = np.floor(len(I)*np.random.rand())
            n = I[int(i)]
        return n

    def is_soma(self):
        if(self.n_node == self.n_soma):
            return True
        else:
            return False

    def set_nodes_values(self):
        i = 0
        for n in self.nodes_list:
            n.xyz = self.location[:,i]
            n.r = self.diameter[i]
            i += 1

    def _set_showing_hist_legends(self):
        self._show_hist = {
        'branch angle': np.arange(0,np.pi,np.pi/20),
        'side branch angle': np.arange(0,np.pi,np.pi/20),
        'distance from root': 30, # np.arange(0,1500,10),
        'global angle': np.arange(0,np.pi,np.pi/20),
        'local angle': np.arange(0,np.pi,np.pi/20),
        'curvature' : np.arange(0,np.pi,np.pi/20),
        'neural important' : np.arange(0,600,10),
        'neural/euclidian_important': np.arange(1,3,.05),
        'euclidian_neuronal' : np.arange(1,3,.05),
        'distance from parent' : 30, # np.arange(0,60,1),
        #'x_location': 30, # np.arange(0,1500,5),
        #'y_location': 30, # np.arange(0,1500,5),
        #'z_location': 30, # np.arange(0,1500,5),
        }
        self._show_title = {
        'branch_angle': 'Angle at the branching points',
        'side_branch_angle': 'Two Side Angles at the Branching Points',
        'distance_from_root': 'Distance from Soma',
        'global_angle': 'Global Angle',
        'local_angle': 'Local Angle',
        'curvature' : 'Curvature',
        'neural_important' : 'Neural Length of Segments',
        'ratio_neural_euclidian_important': 'Neural/Euclidian for Segments',
        'ratio_euclidian_neuronal' : 'Neuronal/Euclidian from All Nodes',
        'distance_from_parent' :'Distance from Parent',
        #'x_location': 'x_location',
        #'y_location': 'y_location',
        #'z_location': 'z_location',
        }
        self._show_legend_x = {
        'branch_angle': 'angle (radian)',
        'side_branch_angle': 'angle (radian)',
        'distance_from_root': 'distance (um)',
        'global_angle': 'angle (radian)',
        'local_angle': 'angle (radian)',
        'curvature' : 'angle (radian)',
        'neural_important' : 'distance (um)',
        'ratio_neural_euclidian_important': 'without unit',
        'ratio_euclidian_neuronal' : 'without unit',
        'distance_from_parent' :'distance (um)',
        #'x_location': 'distance (um)',
        #'y_location': 'distance (um)',
        #'z_location': 'distance (um)',
        }
        self._show_legend_y = {
        'branch_angle': 'density',
        'side_branch_angle': 'density',
        'distance_from_root': 'density',
        'global_angle': 'density',
        'local_angle': 'density',
        'curvature' : 'density',
        'neural_important' : 'density',
        'ratio_neural_euclidian_important': 'density',
        'ratio_euclidian_neuronal' : 'density',
        'distance_from_parent' : 'density',
        #'x_location': 'density',
        #'y_location': 'density',
        #'z_location': 'density',
        }

    def show_features(self, size_x=15, size_y=17):
        self._set_showing_hist_legends()
        m = 2
        n = int(len(self._show_hist.keys())/2) + 1
        k = 1
        plt.figure(figsize=(size_x, size_y))
        for name in self._show_hist.keys():
            plt.subplot(n, m, k)
            print name
            a = self.features[name]
            b = plt.hist(a[~np.isnan(a)], bins=self._show_hist[name], color='g')
            #plt.xlabel(self._show_legend_x[name])
            plt.ylabel(self._show_legend_y[name])
            plt.title(self._show_title[name])
            k += 1

        # plt.subplot(n,m,10)
        # ind = np.arange(4)
        # width = 0.35
        # plt.bar(ind,(self.n_node,self.features['Nbranch'],self.features['initial_segments'],self.features['discrepancy_space']),color='r');
        # #plt.title('Numberical features')
        # #plt.set_xticks(ind + width)
        # plt.xticks(ind,('Nnodes', 'Nbranch', 'Ninitials', 'discrepancy'))


