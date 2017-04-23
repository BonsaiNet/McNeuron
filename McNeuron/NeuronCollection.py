"""Collection of functions for working with a database of neuron."""

import os
import numpy as np
from McNeuron import Neuron
import subsample
import visualize
from copy import deepcopy
import dis_util
from copy import deepcopy
import os
import scipy.io

def load_data(path):
    data = {'X':[], 'Y':[]}
    loc1 = path + "swc1.mat"
    mat1 = scipy.io.loadmat(loc1)
    loc2 = path + "swc2.mat"
    mat2 = scipy.io.loadmat(loc2)
    loc3 = path + "swc3.mat"
    mat3 = scipy.io.loadmat(loc3)
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

    data['Y'] = deepcopy(labels)
    for k in range(59733):
        X = meta['B'][0, k]
        for i in range(X.shape[0]):
            if(X.shape[1]!=0):
                x = X[i,0]
                if(len(x)!=0):
                    name = str(X[i,0][0])[:-2]
                    if(name!='Note' and name!= 'Measuremen'):
                        data['Y'][name].append(str(meta['B'][0, k][i,1][0]))

        X = measures['B'][0, k]
        if(X.shape[1]!=1):
            for i in range(X.shape[0]):
                if(X.shape[1]!=0):
                    x = X[i,0]
                    if(len(x)!=0):
                        name = str(X[i,0][0])[:-2]
                        if(name!='Note' and name!= 'Measuremen'):
                            data['Y'][name].append(str(measures['B'][0, k][i,1][0]))
        else:
            x = X[0,0]
            if(len(x)!=0):
                name = str(X[0,0][0])[:-2]
                if(name!='Note' and name!= 'Measuremen'):
                    data['Y'][name].append(str(measures['B'][0, k][1,0][0]))


        for name in labels:
            if len(data['Y'][name])!=k+1:
                data['Y'][name].append('NaN')
    data['Y']['Description on swc file'] = []
    for i in range(59733):
        data['Y']['Description on swc file'].append(str(texts['B'][0,1][0]))
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
        index += 1
        if(n == name):
            sub_data['X'].append(deepcopy(data['X'][i]))
            for label in data['Y'].keys():
                sub_data['Y'][label].append(deepcopy(data['Y'][label][i]))
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
        self.n_subsampling = 200

    def fit(self, input_format = None, input_file = None):
        if input_format == 'neuron':
            self.database = input_file
        if input_format == 'swc' or input_format == 'Matrix of swc':
            self.read_file(input_format, input_file)

    def set_subsampling(self, number):
        self.n_subsampling = number

    def read_file(self, input_format, input_file):
        self.database = []
        for neuron in input_file:
            n = Neuron(input_format=input_format, input_file=neuron)
            purne_n, dis = subsample.straight_subsample_with_fixed_number(n, self.n_subsampling)
            self.database.append(purne_n)


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
        self.value_all = {}
        self.mean_value = {}
        self.std_value = {}
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
        self.hist_all = {}
        self.mean_hist = {}
        self.std_hist = {}
        for name in list_features.keys():
            for n in range(len(self.database)):
                f = self.features[name][n]
                f = f[~np.isnan(f)]
                hist_fea = \
                    np.histogram(f, bins=hist_range[name])[0].astype(float)
                if(sum(hist_fea) != 0):
                    hist_fea = hist_fea/sum(hist_fea)
                if(n == 0):
                    self.hist_all[name] = np.zeros([len(self.database),
                                                    hist_range[name].shape[0]-1])
                self.hist_all[name][n, :] = hist_fea
            self.mean_hist[name] = self.hist_all[name].mean(axis=0)
            self.std_hist[name] = self.hist_all[name].std(axis=0)
            dis_hist = np.zeros(len(self.database))
            for n in range(len(self.database)):
                a = self.hist_all[name][n, :] - self.mean_hist[name]
                dis_hist[n] = (np.sqrt((a**2).sum()))

            self.std_hist[name] = dis_hist.mean()

    def set_vec_value(self, list_features):
        self.vec_value_all = {}
        self.mean_vec_value = {}
        self.std_vec_value = {}
        for name in list_features:
            for n in range(len(self.database)):
                f = self.features[name][n]
                if(n == 0):
                    self.vec_value_all[name] = np.zeros([len(self.database),
                                                         f.shape[0]])
                self.vec_value_all[name][n, :] = f
            self.mean_vec_value[name] = self.vec_value_all[name].mean(axis=0)
            dis_vec_value = np.zeros(len(self.database))
            for n in range(len(self.database)):
                a = self.vec_value_all[name][n, :] - self.mean_vec_value[name]
                dis_vec_value[n] = (np.sqrt((a**2).sum()))

            self.std_vec_value[name] = dis_vec_value.mean()

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
