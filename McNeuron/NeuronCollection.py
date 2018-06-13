"""Collection of functions for working with a database of neuron."""

import os
import numpy as np
from copy import deepcopy
import os
import scipy.io
import pickle
import McNeuron.Neuron
import McNeuron.subsample
import McNeuron.visualize
import McNeuron.dis_util

def get_all_labels(path):
    loc4 = path + "text.mat"
    texts = scipy.io.loadmat(loc4)
    loc5 = path + "metadata.mat"
    meta = scipy.io.loadmat(loc5)
    loc6 = path + "measures.mat"
    measures = scipy.io.loadmat(loc6)
    labels = {}

    for k in range(59733):
        X = measures['B'][0,k]
        for i in range(X.shape[0]):
            if(X.shape[1]!=0):
                x = X[i,0]
                if(len(x)!=0):
                    name = str(X[i,0][0])[:-2]
                    if(~(name in labels.keys())):
                        if(name!='Note' and name!= 'Measuremen'):
                            labels[name] = []
    for k in range(59733):
        X = meta['B'][0,k]
        for i in range(X.shape[0]):
            if(X.shape[1]!=0):
                x = X[i,0]
                if(len(x)!=0):
                    name = str(X[i,0][0])[:-2]
                    if(~(name in labels.keys())):
                        if(name!='Note' and name!= 'Measuremen'):
                            labels[name] = []
    return labels

def fill_labels(path):
    loc4 = path + "text.mat"
    texts = scipy.io.loadmat(loc4)
    loc5 = path + "metadata.mat"
    meta = scipy.io.loadmat(loc5)
    loc6 = path + "measures.mat"
    measures = scipy.io.loadmat(loc6)
    labels ={'Age Scale': [],
             'Archive Name': [],
             'Average Bifurcation Angle Local': [],
             'Average Bifurcation Angle Remote': [],
             'Average Contraction': [],
             'Average Diameter': [],
             "Average Rall's Ratio": [],
             'Date of Deposition': [],
             'Date of Upload': [],
             'Details about selected neur': [],
             'Development': [],
             'Experiment Protocol': [],
             'Experimental Condition': [],
             'Fractal Dimension': [],
             'Gender': [],
             'Magnification': [],
             'Max Age': [],
             'Max Branch Order': [],
             'Max Euclidean Distance': [],
             'Max Path Distance': [],
             'Max Weight': [],
             'Min Age': [],
             'Min Weight': [],
             'Morphological Attributes': [],
             'NeuroMorpho.Org ID': [],
             'Neuron Name': [],
             'Number of Bifurcations': [],
             'Number of Branches': [],
             'Number of Stems': [],
             'Objective Type': [],
             'Original Format': [],
             'Overall Depth': [],
             'Overall Height': [],
             'Overall Width': [],
             'Partition Asymmetry': [],
             'Physical Integrity': [],
             'Primary Brain Region': [],
             'Primary Cell Class': [],
             'Reconstruction Method': [],
             'Secondary Brain Region': [],
             'Secondary Cell Class': [],
             'Slice Thickness': [],
             'Slicing Direction': [],
             'Soma Surface': [],
             'Species Name': [],
             'Staining Method': [],
             'Strain': [],
             'Structural Domains': [],
             'Tertiary Brain Region': [],
             'Tertiary Cell Class': [],
             'Tissue Shrinkage': [],
             'Total Fragmentation': [],
             'Total Length': [],
             'Total Surface': [],
             'Total Volume': []}

    all_label = labels
    for k in range(59733):
        X = meta['B'][0, k]
        for i in range(X.shape[0]):
            if(X.shape[1] != 0):
                x = X[i, 0]
                if(len(x) != 0):
                    name = str(X[i, 0][0])[:-2]
                    if(name != 'Note' and name != 'Measuremen'):
                        all_label[name].append(str(meta['B'][0, k][i, 1][0]))

        X = measures['B'][0, k]
        if(X.shape[1] != 1):
            for i in range(X.shape[0]):
                if(X.shape[1] != 0):
                    x = X[i, 0]
                    if(len(x) != 0):
                        name = str(X[i,0][0])[:-2]
                        if(name != 'Note' and name != 'Measuremen'):
                            all_label[name].append(str(measures['B'][0, k][i, 1][0]))
        else:
            x = X[0, 0]
            if(len(x) != 0):
                name = str(X[0,0][0])[:-2]
                if(name != 'Note' and name != 'Measuremen'):
                    all_label[name].append(str(measures['B'][0, k][1,0][0]))

        for name in labels:
            if len(all_label[name]) != k+1:
                all_label[name].append('NaN')
    all_label['Description on swc file'] = []
    for i in range(59733):
        all_label['Description on swc file'].append(str(texts['B'][0,1][0]))

def load_data(path, with_swc=True):
    data = {'X':[], 'Y':[]}
    data['Y'] = pickle.load(open(path + "label_all.p", "rb"))
    if with_swc:
        loc1 = path + "swc1.mat"
        mat1 = scipy.io.loadmat(loc1)
        loc2 = path + "swc2.mat"
        mat2 = scipy.io.loadmat(loc2)
        loc3 = path + "swc3.mat"
        mat3 = scipy.io.loadmat(loc3)
        for i in range(20000):
            data['X'].append(mat1['A'][0,i])
        for i in range(20000):
            data['X'].append(mat2['A'][0,i])
        for i in range(19733):
            data['X'].append(mat3['A'][0,i])
    return data

def make_sub_data(data, label, name):
    sub_data = {}
    sub_data['X'] = []
    sub_data['Y'] = {}
    for st in data['Y'].keys():
        sub_data['Y'][st] = []
    index = 0
    for n in data['Y'][label]:
        if(n == name):
            sub_data['X'].append(deepcopy(data['X'][index]))
            for label in data['Y'].keys():
                sub_data['Y'][label].append(deepcopy(data['Y'][label][index]))
        index += 1
    return sub_data


def get_all_path(directory):
    """
    Collecting all the addresses of swc files in the given directory.

    Parameters
    ----------
    directory: str
        The address of the folder

    Returns
    -------
    fileSet: list
        list of all the addresses of *.swc neurons.
    """
    fileSet = []
    for root, dirs, files in os.walk(directory):
        for fileName in files:
            if(fileName[-3:] == 'swc'):
                fileSet.append(directory + root.replace(directory, "")
                               + os.sep + fileName)
    return fileSet

class Collection(object):
    def __init__(self):
        self.value_all = {}
        self.mean_value = {}
        self.std_value = {}
        self.hist_all = {}
        self.mean_hist = {}
        self.std_hist = {}
        self.vec_value_all = {}
        self.mean_vec_value = {}
        self.std_vec_value = {}
        self.database = []
        self.n_database = 0
        self.n_subsampling = 200
        self.len_subsampling = 10

    def fit(self, input_format = 'Matrix of swc', input_file = None):
        if input_format == 'neuron':
            self.database = input_file
        else:
            self.read_file(input_format=input_format,
                           input_file=input_file)
        self.n_database = len(self.database)

    def read_file(self, input_format, input_file):
        if(input_format == 'Matrix of swc'):
            self.index = np.array([])
            index = -1
            for neuron in input_file:
                try:
                    index += 1
                    m = subsample.fast_straigthen_subsample_swc(neuron, self.len_subsampling)
                    n = Neuron(input_file=m)
                    n.set_features()
                    self.index = np.append(self.index, index)
                    self.database.append(n)
                    print(index)
                except:
                    print('ERROR IN:')
                    print(index)

#         if(input_format == 'Matrix of swc without Node class'):
#             self.index = np.array([])
#             index = -1
#             for neuron in input_file:
#                 try:
#                     index += 1
#                     try:
#                         m = subsample.fast_straigthen_subsample_swc(neuron, self.len_subsampling)
#                     except:
#                         neuron = subsample.remove_tip_for_3forks(neuron)
#                         m = subsample.fast_straigthen_subsample_swc(neuron, self.len_subsampling)
#                     n = Neuron(input_format=input_format, input_file=m)
#                     self.index = np.append(self.index, index)
#                     self.database.append(n)
#                     print index
#                 except:
#                     print('ERROR IN:')
#                     print index

#         if(input_format == 'swc'):
#             for neuron in input_file:
#                 n = Neuron(input_format=input_format, input_file=neuron)
#                 purne_n, dis = subsample.straight_subsample_with_fixed_number(n, self.n_subsampling)
#                 self.database.append(purne_n)


    def set_features(self):
        """
        set the range of histogram for each feature.

        hist_range : dict
        ----------
            dictionary of all feature and thier range of histogram.
        """
        self.features = {}
        for name in self.database[0].features.keys():
            self.features[name] = []
            for i in range(len(self.database)):
                self.features[name].append(self.database[i].features[name])

    def set_value(self, list_features):
        for name in list_features:
            for n in range(len(self.database)):
                f = self.features[name][n]
                f = f.mean()
                if(n == 0):
                    self.value_all[name] = np.zeros([len(self.database)])
                self.value_all[name][n] = f
            self.mean_value[name] = self.value_all[name].mean(axis=0)
            self.std_value[name] = self.value_all[name].std(axis=0)

    def set_hist(self, list_features, hist_range):
        for name in list_features.keys():
            self.hist_all[name] = np.zeros([self.n_database,
                                            hist_range[name].shape[0]-1])
            for n in range(self.n_database):
                f = self.features[name][n]
                f = f[~np.isnan(f)]
                hist_fea = \
                    np.histogram(f, bins=hist_range[name])[0].astype(float)
                if(sum(hist_fea) != 0):
                    hist_fea = hist_fea/sum(hist_fea)
                else:
                    hist_fea = np.ones(len(hist_fea))/float(len(hist_fea))

                self.hist_all[name][n, :] = hist_fea
            self.mean_hist[name] = self.hist_all[name].mean(axis=0)
            dis_hist = np.zeros(len(self.database))
            for n in range(len(self.database)):
                a = self.hist_all[name][n, :] - self.mean_hist[name]
                dis_hist[n] = (np.sqrt((a**2).sum()))
            self.std_hist[name] = self.hist_all[name].std(axis=0)
            self.std_hist[name][np.where(self.std_hist[name] == 0)[0]] = \
                self.std_hist[name].mean()
            self.std_hist[name] = dis_hist.mean()

    def set_vec_value(self, list_features):
        for name in list_features:
            for n in range(len(self.database)):
                f = self.features[name][n]
                if(n == 0):
                    self.vec_value_all[name] = np.zeros([len(self.database),
                                                         self.longest_feature(name)])
                self.vec_value_all[name][n, :f.shape[0]] = f
            self.mean_vec_value[name] = self.vec_value_all[name].mean(axis=0)
            dis_vec_value = np.zeros(len(self.database))
            for n in range(len(self.database)):
                a = self.vec_value_all[name][n, :] - self.mean_vec_value[name]
                dis_vec_value[n] = (np.sqrt((a**2).sum()))
            self.std_vec_value[name] = self.vec_value_all[name].std(axis=0)
            self.std_vec_value[name][np.where(self.std_vec_value[name] == 0)[0]] = \
                self.std_vec_value[name].mean()
            self.std_vec_value[name] = dis_vec_value.mean()

    def longest_feature(self, name):
        I = np.zeros(len(self.database))
        for n in range(len(self.database)):
            I[n] = len(self.features[name][n])
        return int(I.max())

    def avoid_zero_std(self, value):
        # for name in self.std_vec_value.keys():
        #     a = self.std_vec_value[name]
        #     (b,) = np.where(a == 0)
        #     a[b] = value
        #     self.std_hist[name] = a
        # for name in self.std_hist.keys():
        #     a = self.std_hist[name]
        #     (b,) = np.where(a == 0)
        #     a[b] = value
        #     self.std_hist[name] = a
        print('done!')

    def normlizor(self, scale):
        dis = []
        for i in range(len(self.database)):
            neuron = self.database[i]
            a = dis_util.distance_from_database_with_name(neuron, self)
            dis.append(a)

        normlizer = dis[0]
        key = dis[0].keys()
        for i in range(1, len(dis)):
            for name in key:
                normlizer[name] = normlizer[name] + dis[i][name]
        for name in key:
            normlizer[name] = normlizer[name]/len(dis)
        for name in key:
            normlizer[name] = scale/normlizer[name]
        return normlizer

    def len_feature(self):
        l = 0
        for name in self.features.keys():
            if(name in self.value_all.keys()):
                l += 1
            if(name in self.hist_all.keys()):
                l += self.hist_all[name].shape[1]
            if(name in self.vec_value_all.keys()):
                l += self.vec_value_all[name].shape[1]
        return l

    def matrix_feature(self):
        M = np.zeros([len(self.database), self.len_feature()])
        index = 0
        for name in self.features.keys():
            if(name in self.value_all.keys()):
                M[:, index:index+1] = np.expand_dims(self.value_all[name], axis=1)
                index = index + 1
            if(name in self.hist_all.keys()):
                l = self.hist_all[name].shape[1]
                M[:, index:index+l] = self.hist_all[name]
                index = index + l
            if(name in self.vec_value_all.keys()):
                l = self.vec_value_all[name].shape[1]
                M[:, index:index+l] = self.vec_value_all[name]
                index = index + l
        return M
