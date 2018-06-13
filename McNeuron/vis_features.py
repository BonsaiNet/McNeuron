from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import McNeuron

def features_attr(feature, feature_type,
                  title,
                  x_title='',
                  y_title='', 
                  hist_range='',
                  x_range='',
                  dim=1):
    feature_list = {'x': feature,
                    'type': feature_type,
                    'title': title,
                    'xlabel': x_title,
                    'ylabel': y_title,
                    'y title': y_title}
    if feature_type == 'histogram':
        feature_list['bins'] = hist_range
    if feature_type == 'density':
        feature_list['range'] = x_range
        feature_list['dim'] = dim
    return feature_list

def feature_ax(feature_list, ax):
    if feature_list['type'] == 'scalar':
        print(feature_list['title']+': '+str(feature_list['x'][0])) 
    else:
        if feature_list['type'] == 'histogram':
            ax.hist(feature_list['x'], feature_list['bins'])
        elif feature_list['type'] == 'density':
            if feature_list['dim'] == 1:
                ax.plot(feature_list['range'], feature_list['x'])
            elif feature_list['dim'] == 2:
                #print feature_list['x'].shape
                ax.imshow(np.transpose(np.reshape(feature_list['x'],[10,10])))
        ax.set_title(feature_list['title'])
        ax.set_xlabel(feature_list['xlabel'])
        ax.set_ylabel(feature_list['ylabel'])    

def set_feature_to_show(neuron):
    list_features = {}
    names = neuron.features.keys()
    for name in names:
        if(name in ['all non trivial initials', 
                    'initial segments',
                    'mean segmental neural length',
                    'initial with branch',
                    'die die',
                    'mean segmental euclidean length',
                    'branch die',
                    'mean segmental neuronal/euclidean',
                    'Nbranch',
                    'branch branch',
                    'Nsoma',
                    'Npassnode',
                    'Nnodes',
                    'asymmetric ratio',
                    'mean neuronal/euclidean'] ):
            list_features[name]=\
            features_attr(feature=neuron.features[name], 
                          title=name,
                          feature_type='scalar')
        elif(name in ['pictural image xy',
                    'pictural image xy tips']):
            list_features[name]=\
                features_attr(feature=neuron.features[name], 
                              title=name,
                              feature_type='density',
                              dim=2)
        elif(name in ['branch depth',
                      'dead depth',
                      'continue depth',
                      'main branch depth',
                   'continue depth'
                   'dead depth'               
                   'main branch depth',
                   'main dead depth',
                   'branch die depth',
                   'branch branch depth',
                   'die die depth']):
            list_features[name]=\
                features_attr(feature=neuron.features[name],
                              title=name,
                              feature_type='density',
                              x_range=np.arange(len(neuron.features[name])),
                              x_title='topological depth',
                              y_title='density',
                              dim=1)
        elif(name in ['global angle',
                       'local angle',
                       'branch angle',
                       'side branch angle'
                       'curvature',
                       'segmental branch angle',
                       'side branch angle']):
            list_features[name]=\
                features_attr(feature=neuron.features[name], 
                              title=name,
                              feature_type='histogram',
                              hist_range=np.arange(0,np.pi,np.pi/20),
                              x_title='angle',
                          y_title='density')
        elif(name == 'distance from parent'):
            list_features[name]=\
                features_attr(feature=neuron.features[name], 
                              title=name,
                              feature_type='histogram',
                              hist_range=np.arange(0,20,1),
                              x_title='angle',
                              y_title='density') 
                
        elif(name == 'distance from root'):
            list_features[name]=\
                features_attr(feature=neuron.features[name], 
                              title=name,
                              feature_type='histogram',
                              hist_range=np.arange(0,500,25),
                              x_title='angle',
                              y_title='density') 
        elif(name in ['segmental neural length',
                   'segmental euclidean length',
                   'curvature']):
            list_features[name]=\
                features_attr(feature=neuron.features[name], 
                              title=name,
                              feature_type='histogram',
                              hist_range=np.arange(1,30,1),
                              x_title='length',
                              y_title='density')
        elif(name in ['neuronal/euclidean for segments',
                      'neuronal/euclidean',
                   ]):
            list_features[name]=\
                features_attr(feature=neuron.features[name], 
                              title=name,
                              feature_type='histogram',
                              hist_range=np.arange(1,3,.05),
                          x_title='ratio',
                          y_title='density')
        elif (name in ['discrepancy space']):
            list_features[name]=\
                features_attr(feature=neuron.features[name], 
                              title=name,
                          feature_type='density',
                          x_range=np.arange(len(neuron.features[name])),
                          x_title='mesh',
                          y_title='density')
                
        elif (name in ['self avoidance']):
            list_features[name]=\
                features_attr(feature=neuron.features[name], 
                              title=name,
                          feature_type='density',
                          x_range=np.arange(len(neuron.features[name])),
                          x_title='mesh',
                          y_title='density')
                
        else:
            print('\n'+'the feature '+name+' is not showing.')
    return list_features
    
def show_features(neuron,
                  fig_x_size=10,
                  fig_y_size=30,
                  x_grid=10,
                  y_grid=4):

    fig = plt.figure(figsize=(fig_x_size, fig_y_size)) 
    gs = gridspec.GridSpec(x_grid, y_grid) 
    index = 1
    McNeuron.visualize.plot_2D(neuron, pass_ax=True,ax=plt.subplot(gs[0]))
    list_features = set_feature_to_show(neuron)
    for name in list_features:
        if list_features[name]['type'] != 'scalar':
            feature_ax(list_features[name], ax=plt.subplot(gs[index]))
            index += 1
        else:
            feature_ax(list_features[name], 1)
    plt.tight_layout()
    plt.show()

    

##########################
#### CUT FROM NEURON #####

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
        print(name)
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
