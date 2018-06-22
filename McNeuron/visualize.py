"""Basic visualization of neurite morphologies using matplotlib."""
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.animation as animation
import pylab as pl
from PIL import Image
from numpy.linalg import inv
import matplotlib
from matplotlib import collections as mc
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import itertools
import math
from copy import deepcopy
import McNeuron.tree_util
import McNeuron.Neuron
import McNeuron.subsample

sys.setrecursionlimit(10000)

def get_2d_image(path, size, dpi, background, show_width):

    neu = Neuron(file_format = 'swc without attributes', input_file=path)
    depth = neu.location[2,:]
    p = neu.location[0:2,:]
    widths= 5*neu.diameter
    widths[0:3] = 0
    m = min(depth)
    M = max(depth)
    depth =  background * ((depth - m)/(M-m))
    colors = []
    lines = []
    patches = []

    for i in range(neu.n_soma):
        x1 = neu.location[0,i]
        y1 = neu.location[1,i]
        r = 1*neu.diameter[i]
        circle = Circle((x1, y1), r, color = str(depth[i]), ec = 'none',fc = 'none')
        patches.append(circle)

    pa = PatchCollection(patches, cmap=matplotlib.cm.gray)
    pa.set_array(depth[0]*np.zeros(neu.n_soma))

    for i in range(len(neu.nodes_list)):
        colors.append(str(depth[i]))
        j = neu.parent_index[i]
        lines.append([(p[0,i],p[1,i]),(p[0,j],p[1,j])])
    if(show_width):
        lc = mc.LineCollection(lines, colors=colors, linewidths = widths)
    else:
        lc = mc.LineCollection(lines, colors=colors)

    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.add_collection(pa)
    fig.set_size_inches([size + 1, size + 1])
    fig.set_dpi(dpi)
    plt.axis('off')
    plt.xlim((min(p[0,:]),max(p[0,:])))
    plt.ylim((min(p[1,:]),max(p[1,:])))
    plt.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    border = (dpi/2)
    return np.squeeze(data[border:-border,border:-border,0])


def projection_on_plane(neuron,
                        orthogonal_vec=np.array([0, 0, 1]),
                        rotation=np.array([0.0])):
    """
    Parameters
    ----------
    neuron: Neuron
        the neuron object to be plotted.
    orthogonal_vec: numpy array
        the orthogonal direction for projecting.

    return
    ------

    dependency
    ----------
    This function needs following data from neuron:
        location
        diameter
        parent_index
    """
    # projection all the nodes on the plane and finding the right pixel for their centers
    normal_1 = orthogonal_vec/np.sqrt(sum(orthogonal_vec**2))
    normal_2 = np.zeros(3)
    normal_2[1] = 1
    A = neuron_util.get_swc_matrix(neuron)

    new_neuron = Neuon()

    new_neuron.location = 1
    image = np.zeros(resolution)
    shift = resolution[0]/2
    normal_vec1 = np.array([0,0,1])
    normal_vec2 = np.array([0,1,0])
    P = project_points(neuron.location, normal_vec1, normal_vec2)

    for n in neuron.nodes_list:
        if(n.parent != None):
            n1, n2, dis = project_point(n, normal_vec1, normal_vec2)
            pix1 = np.floor(n1/gap) + shift
            pix2 = np.floor(n2/gap) + shift
            if(0 <= pix1 and 0 <= pix2 and pix1<resolution[0] and pix2 < resolution[1]):
                image[pix1, pix2] = dis
    return image

def project_points(location, normal_vectors):
    """
    Parameters
    ----------
    normal_vectors : array of shape [2,3]
        Each row should a normal vector and both of them should be orthogonal.

    location : array of shape [3, n_nodes]
        the location of n_nodes number of points

    Returns
    -------
    cordinates: array of shape [2, n_nodes]
        The cordinates of the location on the plane defined by the normal vectors.
    """
    cordinates = np.dot(normal_vectors, location)
    return cordinates

def depth_points(location, orthogonal_vector):
    """
    Parameters
    ----------
    orthogonal_vector : array of shape [3]
        orthogonal_vector that define the plane

    location : array of shape [3, n_nodes]
        the location of n_nodes number of points

    Returns
    -------
    depth: array of shape [n_nodes]
        The depth of the cordinates when they project on the plane.
    """
    depth = np.dot(orthogonal_vector, location)
    return depth

def make_image(neuron, A, scale_depth, index_neuron):
    normal_vectors = A[0:2,:]
    orthogonal_vector = A[2,:]
    depth = depth_points(neuron.location, orthogonal_vector)
    p = project_points(neuron.location, normal_vectors)
    m = min(depth)
    M = max(depth)
    depth =  scale_depth * ((depth - m)/(M-m))
    colors = []
    lines = []
    for i in range(len(neuron.nodes_list)):
        colors.append((depth[i],depth[i],depth[i],1))
        j = neuron.parent_index[i]
        lines.append([(p[0,i],p[1,i]),(p[0,j],p[1,j])])
    lc = mc.LineCollection(lines, colors=colors, linewidths=2)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    pl.axis('off')
    pl.xlim((min(p[0,:]),max(p[0,:])))
    pl.ylim((min(p[1,:]),max(p[1,:])))
    Name = "neuron" + str(index_neuron[0]+1) + "resample" + str(index_neuron[1]+1) + "angle" + str(index_neuron[2]+1) + ".png"
    fig.savefig(Name,figsize=(6, 6), dpi=80)
    img = Image.open(Name)
    img.load()
    data = np.asarray( img, dtype="int32" )
    data = data[:,:,0]
    return data


def make_six_matrix(A):
    six = []
    six.append(A[[0,1,2],:])
    six.append(A[[0,2,1],:])
    six.append(A[[1,2,0],:])
    six.append(A[[1,0,2],:])
    six.append(A[[2,0,1],:])
    six.append(A[[2,1,0],:])
    return six

def make_six_images(neuron,scale_depth,neuron_index, kappa):
    #A = random_unitary_basis(kappa)
    A = np.eye(3)
    six = make_six_matrix(A)
    D = []
    for i in range(6):
        a = np.append(neuron_index,i)
        D.append(make_image(neuron, six[i], scale_depth, a))
    return D

def generate_data(path, scale_depth, n_camrea, kappa):
    """
    input
    -----
    path : list
        list of all the pathes of swc. each element of the list should be a string.

    scale_depth : float in the interval [0,1]
        a value to differentiate between the background and gray level in the image.

    n_camera : int
        number of different angles to set the six images. For each angle, six images will be generated (up,down and four sides)

    kappa : float
        The width of the distribution that the angles come from. Large value for kappa results in the angles close to x aixs
        kappa = 1 is equvalent to the random angle.

    output
    ------
    Data : list of length
    """
    Data = []
    for i in range(len(path)):
        print(path[i])
        neuron = Neuron(file_format = 'swc without attributes', input_file=path[i])
        if(len(neuron.nodes_list) != 0):
            for j in range(n_camrea):
                D = np.asarray(make_six_images(neuron, scale_depth, np.array([i,j]), kappa))
                Data.append(D)
    return Data

def plot_2D(neuron,
            show_width=False,
            show_soma=False,
            line_width=1,
            node_index_red_after=-1,
            node_color=[],
            shift=(0, 0),
            scale=(1, 1),
            save=[],
            pass_ax=False,
            axis=[1,0,0],
            rotation=0,
            ax=''):
    """
    Plotting a neuron. 
    
    Parameters:
    -----------
    neuron: numpy or Neuron object
        If it is numpy it should have swc structure.
        
        
    """
    if isinstance(neuron, np.ndarray):
        location = neuron[:,2:5].T
        widths= neuron[:,5]
        parent_index = neuron[:,6] -1
        parent_index[0] = 0
        n_node = neuron.shape[0]
        n_soma = len(np.where(neuron[:,1]==1)[0])
    else:
        location = deepcopy(neuron.location)
        widths= neuron.diameter
        parent_index = neuron.parent_index
        n_node = neuron.n_node
        n_soma = neuron.n_soma

    projection = rotation_matrix(axis=axis,
                                 theta=rotation)
    location = np.dot(projection, location)
    location[0, :] = location[0,:]-min(location[0,:])
    location[1, :] = location[1,:]-min(location[1,:])
    
    colors = []
    lines = []
    patches = []
    
    # Adding width
    linewidths = line_width*np.ones(n_node)
    if show_width:
        linewidths = widths*linewidths

    # Making red after a node
    if node_index_red_after >=0:
        red_index = neuron.connecting_after_node(node_index_red_after)
    else:
        red_index=[]
   
    # Making line for each edge
    for i in range(n_node):
        if len(np.where(red_index==i)[0])>0:
            colors.append('r')
        else:
            colors.append('k')
            
        j = int(parent_index[i])
        lines.append([(location[0,i] + shift[0],
                       location[1,i] + shift[1]),
                      (location[0,j] + shift[0],
                       location[1,j] + shift[1])])
    if len(node_color) > 0:
        colors = node_color
    lc = mc.LineCollection(lines,
                           linewidths=linewidths,
                           color=colors)
    # Making Soma
    for i in range(n_soma):
        x1 = location[0, i] + shift[0]
        y1 = location[1, i] + shift[1]
        r = widths[i]
        circle = Circle((x1, y1),
                        r, 
                        color='b',
                        ec='none', 
                        fc='none')
        patches.append(circle)

    pa = PatchCollection(patches, cmap=matplotlib.cm.gray)
    pa.set_array(widths[0]*np.zeros(n_soma))
    
    if pass_ax is False:
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        if(show_soma):
            ax.add_collection(pa)
        plt.axis('off')
        plt.xlim((-.001, max(location[0,:])+.001))
        plt.ylim((-.001, max(location[1,:])+.001))
    else:
        ax.add_collection(lc)
        ax.axis('off')
        ax.set_xlim((-.01,max(location[0,:])+.01))
        ax.set_ylim((-.01,max(location[1,:])+.01))
            
    if(len(save)!=0):
        plt.savefig(save, format = "eps")
    if pass_ax is False:
        plt.show()


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise
    rotation about the given axis by theta radians.
    
    Credit: https://stackoverflow.com/questions/6802577/
    rotation-of-3d-vector

    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def plot_3D(neuron):
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go
    N=neuron.n_node

    Xe=[]
    Ye=[]
    Ze=[]
    for e in range(1, N):
        parent = neuron.parent_index[e]
        Xe+=[neuron.location[0, e],neuron.location[0, parent], None]
        Ye+=[neuron.location[1, e],neuron.location[1, parent], None]
        Ze+=[neuron.location[2, e],neuron.location[2, parent], None]

    trace1=go.Scatter3d(x=Xe,
                   y=Ye,
                   z=Ze,
                   mode='lines',
                   line=dict(color='rgb(125,125,125)', width=1),
                   hoverinfo='none'
                   )
    trace2=go.Scatter3d(x=np.zeros(N),
                   y=np.zeros(N),
                   z=np.zeros(N),
                   mode='markers',
                   name='actors',
                   marker=dict(symbol='dot',
                                 size=6,
                                 colorscale='Viridis',
                                 line=dict(color='rgb(50,50,50)', width=0.5)
                                 ),
                   hoverinfo='text'
                   )
    axis=dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title=''
              )


    layout = go.Layout(
             width=500,
             height=500,
             showlegend=False,
             scene=dict(
                 xaxis=dict(axis),
                 yaxis=dict(axis),
                 zaxis=dict(axis),
            ),
         margin=dict(
            t=100
        ),
        hovermode='closest',
        annotations=[
               dict(
               showarrow=False,
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(
                size=14
                )
                )
            ],    )

    data=[trace1, trace2]
    fig=go.Figure(data=data, layout=layout)

    return py.iplot(fig)

def plot_evolution_mcmc(mcmc):
    """
    Showing the evolution of MCMC during the operation.
    """
    k = 0
    for n in mcmc.evo:
        print('step %s' % k)
        k += 1
        plot_2D(n)

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
        (ind,) = np.where(matrix[:,ch]==1)
        ind = ind[ind!=ch]
        L.append(matrix[np.ix_(ind,ind)])
    p = np.zeros(len(L))
    for i in range(len(L)):
        p[i] = L[i].shape[0]
    s = np.argsort(p)
    List = []
    for i in range(len(L)):
        List.append(L[s[i]])
    return List

def box(x_min, x_max, y, matrix, line):
    """
    The box region for each node in the tree.

    """
    L = decompose_immediate_children(matrix)
    length = np.zeros(len(L)+1)
    for i in range(1,1+len(L)):
        length[i] = L[i-1].shape[0] + 1

    for i in range(len(L)):
        x_left = x_min + (x_max-x_min)*(sum(length[0:i+1])/sum(length))
        x_right = x_min + (x_max-x_min)*(sum(length[0:i+2])/sum(length))
        line.append([((x_min + x_max)/2., y), ((x_left + x_right)/2., y-1)])
        if(L[i].shape[0] > 0):
            box(x_left, x_right, y-1, L[i], line)
    return line


def plot_dendrogram(neuron,
                    show_all_nodes=False,
                    save=[],
                    pass_ax=False, ax=''):
    """
    Showing the dendogram of the neuron. In the case that neuron is represented
    by a numpy arry, its the parent index of its swc format.

    Parameters
    ----------
    neuron: Neuron or numpy
        in the case that the input is numpy, it should be swc parent index (start from 1 and
        without 0 in the whole array) and also the initial value should be 0.
    show_all_nodes: boolean
        if Ture, it will show all the nodes, otherwise only the main points are
        taking into account.
    save: str
        the path to save the figure.
    """
    # for swc format: neuron[:,-1]
    if isinstance(neuron, np.ndarray):
        a = neuron
    else:
        a = neuron.parent_index + 1
        a[0] = 0
        a = a.astype(int)

    A = np.zeros([a.shape[0]+1, a.shape[0]+1])
    A[np.arange(1, a.shape[0]+1), a] = 1
    B = inv(np.eye(a.shape[0]+1) - A)
    l = box(0., 1., 0., B, [])
    min_y = 0
    for i in l:
        min_y = min(min_y, i[1][1])
    lc = mc.LineCollection(l)
    if pass_ax is False:
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        plt.axis('off')
        plt.xlim((0, 1))
        plt.ylim((min_y, 0))
    else:
        ax.add_collection(lc)
        ax.axis('off')
        ax.set_xlim((0, 1))
        ax.set_ylim((min_y, 0))        

    if(len(save) != 0):
        plt.savefig(save, format="eps")
    if pass_ax is False:
        plt.draw()

def show_database(collection):
    for n in collection.database:
        plot_2D(n)
    for name in collection.value_all:
        print(name)
        print(collection.mean_value[name])
        print(collection.std_value[name])
        print('\n')
    for name in collection.hist_features.keys():
        print(name)
        m = collection.mean_hist[name]
        v = collection.std_hist[name]/10.
        plt.fill_between(x=collection.hist_features[name][1:], y1=m+v, y2=m-v)
        plt.show()
    plt.imshow(collection.mean_vec_value['pictural image xy'].reshape(10, 10))
    plt.colorbar()
    plt.show()
    plt.imshow(collection.mean_vec_value['pictural image xy tips'].reshape(10, 10))
    plt.colorbar()
    plt.show()
    for name in collection.vec_value:
        print(name)
        m = collection.mean_vec_value[name]
        v = collection.std_vec_value[name]/10.
        plt.fill_between(x=range(0, len(m)), y1=m+v, y2=m-v)
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float').T / cm.sum(axis=1).astype('float')
        cm = cm.T
        cm = np.floor(cm*100.).astype(int)
        print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)




    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    #plt.xlabel('Predicted label')

def get_segment_collection_neuron(collection,
                                  row=None,
                                  column=None,
                                  scale=None):
    lines = []
    index = 0
    if row is None:
        row = np.floor(np.sqrt(len(collection)))
    if column is None:
        column = np.floor(float(len(collection))/row) + 1
    if scale is None:
        scale = 2.5 * column * 500
    for r in np.arange(row):
        for col in np.arange(column):
            if(index < len(collection)):
                neuron = collection[index]
                index += 1
                p = neuron.location[0:2, :]
                p = p/scale
                for i in range(neuron.n_node):
                    j = neuron.parent_index[i]
                    lines.append([(p[0, i] + col/column, p[1, i] + 1 - r/row),
                                  (p[0, j] + col/column, p[1, j] + 1 - r/row)])
    lc = mc.LineCollection(lines, color='k')
    return lc

def get_segment_collection(neuron):
    lines = []
    p = neuron.location[0:2, :]
    p = p/(.2*p.max())
    for i in range(p.shape[1]):
        j = neuron.parent_index[i]
        if(j>=0):
            lines.append([(p[0,i] ,p[1,i]),
                        (p[0,j],p[1,j] )])
    lc = mc.LineCollection(lines, color = 'k')
    return lc

def evolution_with_increasing_node(neuron):
    for i in range(10):
        m = neuron_util.get_swc_matrix(neuron)
        m = m[40*i:, :]
        I = m[:, 6]
        I = I - 40*i
        I[I < 0] = -1
        m[:, 6] = I
        n = Neuron(input_file=m, input_format='Matrix of swc without Node class')
        lc = get_segment_collection(n)
        fig = plt.figure(figsize=(4, 4))
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        plt.axis('off')
        plt.xlim((-6, 6))
        plt.ylim((-6, 6))
        #plt.draw()
        plt.show()

def nodes_laying_toward_soma(parents, selected_nodes):
    all_nodes = np.array([])
    for i in selected_nodes:
        par = i
        while par != 0:
            all_nodes = np.append(all_nodes, par)
            par = parents[par]
    all_nodes = np.append(all_nodes, 0)
    all_nodes = np.unique(all_nodes).astype(int)
    return all_nodes

def topological_depth(swc_matirx):
    neuron = Neuron(swc_matirx)
    branch_order = tree_util.branch_order(neuron.parent_index)
    distance_from_parent = neuron.distance_from_parent()
    main, parent_main_point, neural, euclidean = \
        neuron.get_neural_and_euclid_lenght_main_point(branch_order, distance_from_parent)
        
    reg_neuron = Neuron(subsample.regular_subsample(swc_matirx))
    topo_depth = np.zeros(swc_matirx.shape[0])
    depth_main = neuron_util.dendogram_depth(reg_neuron.parent_index)
    topo_depth[main] = depth_main
    for i in range(1, swc_matirx.shape[0]):
        b = True
        par = i
        while b:
            (index,) = np.where(main==par)
            if len(index) != 0:
                topo_depth[i] = topo_depth[main[index]]
                b = False
            par = neuron.parent_index[par]
    return topo_depth
            
    
def main_segments(swc_matirx):
    neuron = Neuron(swc_matirx)
    branch_order = tree_util.branch_order(neuron.parent_index)
    distance_from_parent = neuron.distance_from_parent()
    main, parent_main_point, neural, euclidean = \
        neuron.get_neural_and_euclid_lenght_main_point(branch_order, distance_from_parent)
    branch_branch, branch_die, die_die, initial_with_branch = \
        neuron.branching_type(main, parent_main_point) 
    ind_main = nodes_laying_toward_soma(neuron.parent_index,
                                             np.array(np.append(branch_die, die_die)))
    
    main_seg = np.zeros(len(neuron.parent_index))
    main_seg[ind_main] = 1
    return main_seg.astype(int)

def plot_with_color(swc_matirx,
                    color_depth=[],
                    save=[],
                    pass_ax=False, ax=''):
    if len(color_depth)==0:
        color_depth = np.ones(swc_matirx.shape[0])
    p = swc_matirx[:, 2:5]
    m = max(max(p[:, 0]) - min(p[:, 0]), max(p[:, 1]) - min(p[:, 1]))
    p[:, 0] = (p[:, 0]-min(p[:, 0]))/m
    p[:, 1] = (p[:, 1]-min(p[:, 1]))/m
    colors = []
    lines = []
    patches = []

    pa = PatchCollection(patches, cmap=matplotlib.cm.gray)
    colors = []
    colors_binary = main_segments(swc_matirx)
    linewidths=[]
    for i in range(1, swc_matirx.shape[0]):
        j = swc_matirx[i,6] - 1
        lines.append([(p[i,0],p[i,1]),(p[j,0],p[j,1])])
        if colors_binary[i] == 0:
            colors.append((0,0,1))
            linewidths.append(1)
        else:
            colors.append((color_depth[i],0,1-color_depth[i]))
            linewidths.append(2)
            #colors.append((1,0,0))
    lc = mc.LineCollection(colors=colors,segments=lines,linewidths=linewidths)
    if pass_ax is False:
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        fig.set_size_inches([8, 8])
    else:
        ax.add_collection(lc)     
    ax.axis('off')
    m = min(min(p[:,0]),min(p[:,1]))-.001
    ma = max(max(p[:,0]),max(p[:,1])) +.001
    ax.set_xlim((m,ma))
    ax.set_ylim((m,ma))

    if(len(save)!=0):
        plt.savefig(save, format = "eps")
    if pass_ax is False:
        plt.show()


def topological_depth(swc_matirx):
    neuron = Neuron(swc_matirx)
    branch_order = tree_util.branch_order(neuron.parent_index)
    distance_from_parent = neuron.distance_from_parent()
    main, parent_main_point, neural, euclidean = \
        neuron.get_neural_and_euclid_lenght_main_point(branch_order, distance_from_parent)
        
    reg_neuron = Neuron(subsample.regular_subsample(swc_matirx))
    topo_depth = np.zeros(swc_matirx.shape[0])
    depth_main = neuron_util.dendogram_depth(reg_neuron.parent_index)
    topo_depth[main] = depth_main
    for i in range(1, swc_matirx.shape[0]):
        b = True
        par = i
        while b:
            (index,) = np.where(main==par)
            if len(index) != 0:
                topo_depth[i] = topo_depth[main[index]]
                b = False
            par = neuron.parent_index[par]
    return topo_depth
   
def get_dendrogram_as_tree(swc_matrix):
    swc = swc_matrix
    swc_parent = swc_matrix[:, 6].astype(int)
    parent_index = swc_parent
    parent_index[0] = 0
    A = np.zeros([parent_index.shape[0]+1, parent_index.shape[0]+1])
    A[np.arange(1, parent_index.shape[0]+1), parent_index] = 1
    B = inv(np.eye(parent_index.shape[0]+1) - A)
    l = box(0., 1., 0., B, [])
    swc = np.zeros([len(l),7])
    for i in range(len(l)):
        swc[i, 2] = l[i][1][0]
        swc[i, 3] = l[i][1][1]
        if i>0:
            a = np.where((swc[:, 2] - l[i][0][0])**2 + (swc[:, 3] - l[i][0][1])**2==0)[0]
            swc[i, 6] = a+1
    swc[0, 6] = -1
    return swc
