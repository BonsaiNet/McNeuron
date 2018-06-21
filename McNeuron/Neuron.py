"""Neuron class to extract the features of the neuron and perturb it."""
import numpy as np
from numpy import linalg as LA
import math
from scipy.sparse import csr_matrix
from copy import deepcopy
from numpy.linalg import inv
from sklearn import preprocessing
import McNeuron.swc_util as swc_util
import McNeuron.tree_util as tree_util
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# np.random.seed(0)

class Neuron:
    def __init__(self, input_file=None):
        """
        Making an Neuron object by inserting swc txt file or numpy array.

        Parameters:
        -----------
        input_file:
        Retruns:
        --------
        an object with following attributes:
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

    def read_swc_matrix(self, swc_matrix):
        """
        Reads matrix of swc format and asign these attributes:
        n_soma, n_node, nodes_list, location, nodes_type
        diameter, parent_index

        Parameters
        ----------
        input_file: numpy of shape [n, 7]
            .swc format
        """
        self.nodes_type = np.squeeze(swc_matrix[:, 1])
        self.nodes_type = np.array(self.nodes_type, dtype=int)
        self.n_node = swc_matrix.shape[0]
        self.location = swc_matrix[:, 2:5].T
        for i in range(3):
            self.location[i, :] = self.location[i, :] - self.location[i, 0]
        (I,) = np.where(self.nodes_type == 1)
        self.n_soma = len(I)
        self.diameter = np.squeeze(swc_matrix[:, 5])
        self.parent_index = np.squeeze(swc_matrix[:, 6]) - 1
        self.parent_index[0] = 0
        self.parent_index = np.array(self.parent_index, dtype=int)

    def get_swc(self):
        """
        Returning the swc file of the neuron.
        """
        swc = np.zeros([self.n_node,7])
        swc[:, 0] = np.arange(self.n_node)+1
        swc[:, 1] = self.nodes_type
        swc[:, 2] = self.location[0, :]
        swc[:, 3] = self.location[1, :]
        swc[:, 4] = self.location[2, :]
        swc[:, 5] = self.diameter
        swc[:, 6] = self.parent_index + 1
        swc[0, 6] = -1
        return swc

    def __str__(self):
        """
        describtion.
        """
        return "Neuron found with " + str(self.n_node) + " number of nodes and"+ str(self.n_soma) + "number of node representing soma."

    def set_features(self, name):
        """
        The neuron object equip with features based on the name of features. Name can be:
       'toy_mcmc', 'mcmc', 'l_measure', 'motif', 
       
        Parameters:
        -----------
        name: str
            the name of the feature to be added
        """
        self.basic_features()
        if name == 'toy_mcmc':
            self.toy_mcmc_features()
        elif name == 'mcmc':
            self.mcmc_features()
        elif name == 'l_measure':
            self.L_measure_features()
        elif name == 'motif':
            self.motif_features()
        #self.geometrical_features()

    def basic_features(self):
        """
        Returns:
        --------
            branch_order : array of shape = [n_node]
            The number of children of the nodes. It can be and integer number for
            the root (first element) and only 0, 1 or 2 for other nodes.
        """
        self.set_branch_order()
        self.features['Nnodes'] = np.array([self.n_node])
        self.features['Nsoma'] = np.array([self.n_soma])
        (num_branches,) = np.where(self.features['branch order'][self.n_soma:] >= 2)
        self.features['Nbranch'] = np.array([len(num_branches)])
        self.features['initial segments'] = np.array([self.features['branch order'][0]])

    def motif_features(self):
        self.set_branch_order()
        branch_order = self.features['branch order']
        (num_branches,) = np.where(self.features['branch order'][self.n_soma:] == 2)
        (num_dead,) = np.where(self.features['branch order'][self.n_soma:] == 0)
        distance_from_parent = self.distance_from_parent()

        critical_points, parent_critical_points = \
            self.critical_points_and_their_parents(self.parent_index,
                                                   branch_order)
        branch_branch, branch_die, die_die, branching_stems = \
            self.branching_type(critical_points,
                                parent_critical_points)

        branch_depth, continue_depth, dead_depth, \
        branch_branch_depth, branch_die_depth, die_die_depth = \
            self.type_at_depth(branch_order, 
                              self.parent_index, 
                              branch_branch, 
                              branch_die, 
                              die_die)

        self.features['number of branch'] = np.array([len(num_branches)])
        self.features['number of dead'] = np.array([len(num_dead)])
        self.features['branch depth'] = branch_depth
        self.features['continue depth'] = continue_depth
        self.features['dead depth'] = dead_depth
        self.features['branch branch'] = np.array([len(branch_branch)])
        self.features['branch die'] = np.array([len(branch_die)])
        self.features['die die'] = np.array([len(die_die)])
        self.features['branch branch depth'] = branch_branch_depth
        self.features['branch die depth'] = branch_die_depth
        self.features['die die depth'] = die_die_depth
        self.features['branching stems'] = np.array([branching_stems])
        self.features['all non trivial initials'] =\
            np.array([self.all_non_trivial_initials()])
        
        #a = float(self.features['die die']*self.features['branch branch'])
    #         asym = 0
    #         if a != 0:
    #             asym = float((4*self.features['branch die']**2))/a
    #         self.features['asymmetric ratio'] = np.array([asym])
        #(num_pass,) = np.where(self.features['branch order'][self.n_soma:] == 1)
        # self.features['Npassnode'] = np.array([len(num_pass)])


    def branching_type(self, critical_points, parent_critical_points):
        le1 = preprocessing.LabelEncoder()
        le2 = preprocessing.LabelEncoder()

        le1.fit(critical_points)
        parent_critical_points_transform = le1.transform(parent_critical_points)
        branch_critical_point = np.unique(parent_critical_points_transform)
        parent_branch_critical_point = parent_critical_points_transform[branch_critical_point]

        le2.fit(branch_critical_point)
        parent_branch_critical_point_transform = le2.transform(parent_branch_critical_point)
        branch_order_critical_point = tree_util.branch_order(parent_branch_critical_point_transform)

        branch_branch = np.where(branch_order_critical_point[1:] == 2)[0] + 1
        branch_die = np.where(branch_order_critical_point[1:] == 1)[0] + 1
        branch_branch = le1.inverse_transform(le2.inverse_transform(branch_branch))
        branch_die = le1.inverse_transform(le2.inverse_transform(branch_die))
        die_die = np.setxor1d(np.setxor1d(np.unique(parent_critical_points)[1:], 
                                          branch_branch), branch_die)
        branching_stems = branch_order_critical_point[0]
        return branch_branch, branch_die, die_die, branching_stems

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
        path_length = self.neural_distance_from_root(distance_from_parent)
        local_angle = self.local_angle(branch_order)
        global_angle = self.global_angle()
        branch_angle, side_angle = self.branch_angle(branch_order)
        curvature = self.set_curvature()
        #fractal = self.set_discrepancy(np.arange(.01, 5,.01))
        main, parent_critical_point, path_length_critical, euclidean = \
            self.get_neural_and_euclid_lenght_critical_point(branch_order, distance_from_parent)

        self.features['global angle'] = global_angle[self.n_soma:]
        self.features['local angle'] = local_angle[self.n_soma:]
        self.features['distance from parent'] = distance_from_parent[self.n_soma:]
        self.features['distance from root'] = distance_from_root[self.n_soma:]/mean_distance_from_parent
        self.features['path_length/euclidean'] = path_length[self.n_soma:]/distance_from_root[self.n_soma:]
        self.features['mean Contraction'] = \
           np.array([((self.features['path_length/euclidean'] - 1.)).mean()])
        self.features['branch angle'] = branch_angle
        self.features['side branch angle'] = side_angle
        #self.features['curvature'] = curvature
        #self.features['discrepancy space'] = fractal[:,0]
        #self.features['self avoidance'] = fractal[:,1]
        #self.features['pictural image xy'] = self.set_pictural_xy()
        #self.features['pictural image xyz'] = self.set_pictural_xyz(5., 5., 5.)
        #self.features['cylindrical density'] =
        #self.features['pictural image xy tips'] = self.set_pictural_tips(branch_order,10., 10.)
        self.features['segmental neural length'] = path_length_critical/mean_distance_from_parent
        self.features['segmental euclidean length'] = euclidean/mean_distance_from_parent
        self.features['mean segmental neural length'] = \
           np.array([self.features['segmental neural length'].mean()])
        self.features['mean segmental euclidean length'] = \
           np.array([self.features['segmental euclidean length'].mean()])
        #self.features['neuronal/euclidean for segments'] = neural/euclidean
        #self.features['mean segmental neuronal/euclidean'] = \
        #   np.array([np.sqrt(((self.features['neuronal/euclidean for segments'] - 1.)**2).mean())])
        self.features['segmental branch angle'] = \
           self.set_branch_angle_segment(main, parent_critical_point)


    def neural_distance_from_root(self, distance_from_parent):
        a = np.arange(self.n_node)
        dis = np.zeros(self.n_node)
        while(sum(a) != 0):
            dis += distance_from_parent[a]
            a = self.parent_index[a]
        return dis

    def get_neural_and_euclid_lenght_critical_point(self, branch_order, distance_from_parent):
        (critical_points,) = np.where(branch_order[1:] != 1)
        critical_points += 1
        critical_points = np.append(0, critical_points)

        parent_critical_points = np.arange(self.n_node)

        neural = np.zeros(self.n_node)
        for i in range(1, self.n_node):
            if(branch_order[i] == 1):
                neural[i] = neural[self.parent_index[i]] + distance_from_parent[i]
                parent_critical_points[i] = parent_critical_points[self.parent_index[i]]
        neural = neural[self.parent_index[critical_points]][1:] + distance_from_parent[critical_points][1:]
        parent_critical_points = parent_critical_points[self.parent_index[critical_points]]
        euclidean = LA.norm(self.location[:, critical_points] -
                            self.location[:, parent_critical_points], axis=0)[1:]
        return critical_points, parent_critical_points, neural, euclidean


    def get_parent_index_of_subset(self, subset, parent_of_subset):
        le = preprocessing.LabelEncoder()
        le.fit(subset)
        parent = le.transform(parent_of_subset)
        return parent

    def critical_points_and_their_parents(self,
                                          parent_index,
                                          branch_order):
        (critical_points,) = np.where(branch_order[1:] != 1)
        critical_points += 1
        critical_points = np.append(0, critical_points)

        parent_critical_points = np.array([], dtype=int)
        for node in critical_points:
            current_node = parent_index[node]
            while current_node not in critical_points: 
                current_node = parent_index[current_node]
            parent_critical_points = np.append(parent_critical_points,
                                               int(current_node))
        return critical_points, parent_critical_points

    def set_branch_order(self):
        if 'branch order' not in self.features.keys():
            self.features['branch order'] = tree_util.branch_order(self.parent_index)

    def type_at_depth(self,
                      branch_order,
                      parent_index,
                      branch_branch=[],
                      branch_die=[],
                      die_die=[]):
        dead_index = np.where(branch_order == 0)[0]
        continue_index = np.where(branch_order == 1)[0]
        branch_index = np.where(branch_order == 2)[0]
        depth_all = tree_util.dendogram_depth(parent_index).astype(int)
        m = int(depth_all.max()+1)
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
        return branch_depth, continue_depth, dead_depth, \
    branch_branch_depth, branch_die_depth, die_die_depth

    def all_non_trivial_initials(self):
        (I,)=np.where(self.parent_index[self.n_soma:]==0)
        # count=0
        # for i in I:
        #     if i!=0:
        #         (J,)=np.where(self.parent_index == i)
        #         if(len(J) != 0):
        #             count += 1
        # return count
        return len(I)

    def make_fixed_length_vec(self, input_vec, length_vec):
        l = len(input_vec)
        if(l < length_vec):
            fixed_vec = np.zeros(length_vec)
            fixed_vec[0:l] = input_vec
        else:
            fixed_vec = input_vec[:length_vec]
        return fixed_vec

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
            child_index = np.array([list(np.where(self.parent_index==i)[0]) for i in I]).T
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
        child_index = [np.where(self.parent_index == i)[0][0] for i in I]
        dir1 = self.location[:, I] - self.location[:, self.parent_index[I]]
        dir2 = self.location[:, I] - self.location[:, child_index]
        local_angle = self.angle_vec_matrix(dir1, dir2)
        if(len(local_angle) == 0):
            local_angle = np.array([0])
        return local_angle


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



    def set_loc_diam(self):
        self.location = np.zeros([3,self.n_node])
        self.diameter = np.zeros(self.n_node)
        for n in range(self.n_node):
            self.location[:,n] = self.nodes_list[n].xyz
            self.diameter[n] = self.nodes_list[n].r

    def get_connection(self):
        """
        Gets the full connection matrix for neuron. connection is an array with
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
        return connection

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

    def diameter_features(self):
        distance_from_root = self.distance_from_root()
        diameter_euclidean = self.diameter_euclidean(distance_from_root, bins=10)
        self.features['diameter euclidean (bins)'] = diameter_euclidean

    def get_rest_of_neuron_after_node(self, node):
        le = preprocessing.LabelEncoder()
        index = self.connecting_after_node(node)
        le.fit(index)
        le.transform(index) 
        n_node = len(index)
        swc=np.zeros([n_node, 7])
        swc[:, 0] = np.arange(n_node)
        swc[:, 1] = self.nodes_type[index]
        swc[:, 2] = self.location[0, index]
        swc[:, 3] = self.location[1, index]
        swc[:, 4] = self.location[2, index]
        swc[:, 5] = self.diameter[index]
        swc[1:,6] = le.transform(self.parent_index[index[1:]]) + 1
        swc[0, 6] = -1
        return swc

    def l_measure_features(self):
        """
        Measuring all the L-measure features of neuron.
        Ref: https://www.nature.com/articles/nprot.2008.51
        and http://farsight-toolkit.org/wiki/L_Measure_functions

        L-measure consists of following measures:
        """
        soma_indices = np.where(self.node_type==1)[0]
        self.set_branch_order()
        branch_branch, branch_die, die_die, branching_stems = \
            self.branching_type(main, parent_critical_point)
        (num_branches,) = np.where(self.features['branch order'][self.n_soma:] >= 2)
        (tips,) = np.where(self.features['branch order'][self.n_soma:] == 0)

        self.features['Width X'] = max(self.location[0,:]) - min(self.location[0,:])
        self.features['Heigth Y'] = max(self.location[1,:]) - min(self.location[1,:])
        self.features['Depth Z'] = max(self.location[2,:]) - min(self.location[2,:])
        self.features['Soma X Position'] = self.location[0,0]
        self.features['Soma Y Position'] = self.location[1,0]
        self.features['Soma Z Position'] = self.location[2,0]
        self.features['Soma Radii'] = self.diameter[soma_indices].mean()
        ## need to be corrected
        self.features['Soma Surface Area'] = np.power(self.diameter[soma_indices],2).mean()
        self.features['Soma Volume'] = np.power(self.diameter[soma_indices], 3).mean()
        self.features['Skewness X'] = self.location[0,:].mean() - self.location[0,0]
        self.features['Skewness Y'] = self.location[1,:].mean() - self.location[1,0]
        self.features['Skewness Z'] = self.location[2,:].mean() - self.location[2,0]
        self.features['Euclidain Skewness'] = \
            np.sqrt(sum(self.features['Skewness X']**2 + \
            self.features['Skewness Y']**2 + \
            self.features['Skewness Z']))
        self.features['Stems'] = self.branch_order[0]
        self.features['Branching Stems'] = branching_stems
        self.features['Branch Pt'] = np.array([len(num_branches)])
        ## NOT CLEAR
        self.features['Segments'] = 1
        self.features['Tips'] = np.array([len(tips)])
        self.features['Length'] = 1
        self.features['Surface Area'] = 1
        self.features['Volume'] = 1
        self.features['Burke Taper'] = 1
        self.features['Hillman Taper'] = 1
        self.features['Volume'] = 1
        self.features['Average Radius'] = self.diameter.mean()
    
    def toy_mcmc_features(self):
        self.set_branch_order()
        distance_from_parent = self.distance_from_parent()
        distance_from_root = self.distance_from_root()
        (num_branches,) = np.where(self.features['branch order'][self.n_soma:] >= 2)
        path_length = self.neural_distance_from_root(distance_from_parent)
        self.features['path_length/euclidean'] = path_length[self.n_soma:]/distance_from_root[self.n_soma:]
        self.features['mean Contraction'] = \
           np.array([((self.features['path_length/euclidean'] - 1.)).mean()])     
        self.features['Branch Pt'] = np.array([len(num_branches)])
