"""A set of utilities for downloading and reading neuromorpho data"""
import sys
import urllib
import ast
import numpy as np
import pandas as pd
import pickle
import re
import json
from urllib.request import urlopen

#from bs4 import BeautifulSoup
#from unidecode import unidecode
from copy import deepcopy
import math
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from collections import Counter
import operator

def get_dict_from_table(trs):
    from unidecode import unidecode
    table_dict = {}
    for tr in trs:
        if len(tr.find_all('td')) == 2:
            k, v = [t.text for t in tr.find_all('td')]
            k = unidecode(k.replace(':', '').strip())
            v = unidecode(v.replace(':', '').strip())
            table_dict[k] = v
    return table_dict


def find_archive_link(a):
    for a in soup.find_all('a'):
        if a is not None and a.find('input') is not None:
            if 'Link to original archive' == a.find('input').attrs['value']:
                archive_link = a.attrs['href']
    return a

def get_metadata(metadata_html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(metadata_html, 'html.parser')
    table1 = soup.find_all('tbody')[1]
    trs = table1.find_all('tr')[2].find_all('tr')
    d1 = get_dict_from_table(trs)

    
    table2 = soup.find_all('tbody')[4]
    trs = table2.find_all('tr')
    d2 = get_dict_from_table(trs)

    table3 = soup.find_all('tbody')[8]
    trs = table3.find_all('tr')
    d3 = get_dict_from_table(trs)
    
    dic = {}
    dic.update(d1)
    dic.update(d2)
    dic.update(d3)
    return dic

# def swc(swc_html):
#     neuron = np.zeros([1,7])
#     for line in swc_html:
#         line = line.lstrip()
#         #print line
#         if line[0] is not '#' and len(line) > 2:
#             #print line
#             l= '['+re.sub('(\d) +', r'\1,', line[:-1])+']'
#             #print l
#             l= re.sub('(\.) ', r'\1,', l)
#             #print l
#             neuron = np.append(neuron, np.expand_dims(np.array(ast.literal_eval(l)),axis=0),axis=0)
#     return neuron[1:,:]

def swc(swc_html):
    l = swc_html.readlines()
    start = 0
    sharp = 35
    while(l[start][0]==35):
        start +=1
    neuron = np.loadtxt(l[start:])
    return neuron

def download_neuromorpho(start_nmo=1, end_nmo=50):
    """returning the a dataframe (panadas) that each row is a neuron that is
    registered in the neuromorpho.org and the columns are the attributes of 
    neuron (swc, Archive Name,...).
    The data are from neuromorpho.org but there is a backup in gmu website: e.g.
    http://cng.gmu.edu:8080/search/keyword/summary?q=nmo_00001
    
    Parameteres:
    ------------
    start_nmo: int
        the index of first neuron for downloading (neuromorpho indexing)
    end_nmo: int
        the index of last neuron for downloading (neuromorpho indexing)
        
    Retruns:
    --------
    morph_data: Dataframe
        Dataframe of downloaded neurons
    errors: list
        nmo index of neurons that an error raised while downloadig (the error
        might be raised for various reasons; the neuron does not exist, unusual 
        registration format, the link is not vali anymore, ...)
        
    """
    all_neuron = []
    errors = []
    for nmo in range(start_nmo, end_nmo,1):
        try:
            if np.mod(nmo, 100)==0:
                print(nmo)
            dic_all = {}
            neuromorpho_api = urllib.request.urlopen(\
                    "http://neuromorpho.org/api/neuron/id/"+str(nmo)).read()
            neuron_dict = json.loads(neuromorpho_api.decode("utf-8"))
            neuron_name = neuron_dict['neuron_name']
            archive_name = re.sub(' ', '%20', neuron_dict['archive'])

            neuromorpho_link_swc = 'http://neuromorpho.org/dableFiles/'\
                + archive_name.lower() + '/CNG%20version/' + neuron_name \
                + '.CNG.swc'

            swc_html = urllib.request.Request(neuromorpho_link_swc)
            swc_html = urllib.request.urlopen(swc_html)
            neuromorpho_link = \
                'http://neuromorpho.org/neuron_info.jsp?neuron_name='\
                + neuron_name     
            metadata_html = urllib.request.Request(neuromorpho_link)
            metadata_html = str(urllib.request.urlopen(metadata_html).read())
            dic_all = get_metadata(metadata_html)
            dic_all['link in neuromorpho'] = str(neuromorpho_link)
            dic_all['swc'] = swc(swc_html)
            dic_all = {**dic_all, **neuron_dict}
            all_neuron.append(dic_all)
        except:
            errors.append(nmo)
        morph_data = pd.DataFrame(all_neuron)
    return morph_data, errors

def step_save_downloading(path='',
                          saving_neurons = 5000, 
                          n_total_neurons = 90000):
    all_errors = np.array([])
    steps = int(np.floor(n_total_neurons/saving_neurons)+1)
    for i in range(0, steps):
        start_nmo=1+i*saving_neurons
        end_nmo=(i+1)*saving_neurons
        print(start_nmo)
        neurons, errors = \
            download_neuromorpho(start_nmo=start_nmo,
                                                    end_nmo=end_nmo)
        name = path+'neurons_from_'+str(start_nmo)+'_to_'+str(end_nmo)+'.csv'
        neurons.to_pickle(name)
        all_errors = np.append(all_errors, np.array(errors))
    all_errors.to_pickle(path+'errors.csv')
    
def retrive_make_one_file(saving_neurons = 5000, 
                          n_total_neurons = 90000,
                          dump_data_path='./',
                         name_file='save'):
    neurons = pd.DataFrame()

    steps = int(np.floor(n_total_neurons/saving_neurons)+1)
    for i in range(0, steps):
        print(i)
        start_nmo=1+i*saving_neurons
        end_nmo=(i+1)*saving_neurons
        name = 'neurons_from_'+str(start_nmo)+'_to_'+str(end_nmo)+'.csv'
        s = pickle.load(open(dump_data_path+name, 'rb'))
        neurons = neurons.append(s)

    neurons = neurons.reset_index()
    del neurons['index']
    neurons.to_pickle('./'+name_file+'.csv')
    return neurons

def read_data(meta_data_path, swc_path):
    morph_data = pd.read_pickle(meta_data_path)
    morph_data['swc'] = regular_swc = np.load(swc_path)
    return morph_data

def get_all_labels(neurons, labels):
    """
    Listing al the labels and the values in the dataset. i.e. if there
    is 'cell_type' as label and ['type_a', 'type_b'] for an entry, then
    it adds ['cell_type : type_a', 'cell_type : type_b'] to the existing
    list of all possible combinations of labels and values.
    
    Parameters:
    -----------
    neurons: pandas
        columns are lables and the entries can be a string (one values)
        or a list of strings
    
    labels: list
        list of labels form neurons to take into account
    
    Returns:
    --------
    List_all_labels: list
        list of all combinations of labels and values
    """
    List_all_labels = []
    for i in range(neurons.shape[0]):
        if np.mod(i,100)==0:
            print(i)
        for name in labels:
            label = neurons[name][i]
            if isinstance(label, str):
                List_all_labels = List_all_labels + [name + ' : ' + label]
            if isinstance(label, list):
                for lab in label:
                    List_all_labels = List_all_labels + [name + ' : ' + lab]
            List_all_labels = list(set(List_all_labels))
    return List_all_labels

def csr_matrix_labels(neurons, labels, labels_encoder):
    """
    Getting the csr_matrix to easily select the columns.
    
    Parameters:
    -----------
    neurons: panadas
        all neurons with the labels
    
    labels: list
        name of the colums from neurons
    
    labels_encoder: LabelEncoder
        encoded version of all labels+values
    
    Returns:
    --------
    attribute_mat: csr_matrix
        size = (#neurons rows, #labels_encoder)
        it is one iff the corresponding neuron has the values
        at the label. 
    """
    size = neurons.shape[0]
    attribute_mat = csr_matrix((size, len(labels_encoder.classes_)))
    for i in range(size):
        if np.mod(i,100)==0:
            print(i)
        for name in labels:
            label = neurons[name][i]
            if isinstance(label, str):
                index = labels_encoder.transform([name + ' : ' + label])[0]
                attribute_mat[i, index]= 1
            if isinstance(label, list):
                for lab in label:
                    index = labels_encoder.transform([name + ' : ' + lab])[0]
                    attribute_mat[i, index]= 1
    return attribute_mat

def count_label(neurons, label_name):
    """
    Return the value of a labels and number of frequencies in all neurons.
    
    Parameters:
    -----------
    neurons : pandas
        dataframe of all data with keys as the labels
    
    label_name: str
        one of the key(s) in neurons
    
    label_type: list or str
        the type of object in one array of label_name
        
    return:
    -------
    the list of labels with their frequencies
    
    """
    if label_name == 'cell_type' or label_name == 'brain_region':
        MyList  = [tuple(sorted(i)) for i in neurons[label_name] if i is not None]
    else:
        MyList  = list(neurons[label_name])
    stats = dict(Counter(MyList))
    return sorted(stats.items(), key=operator.itemgetter(1))[::-1]

def get_neurons_in_group(attribute_mat,
                         labels_encoder,
                         labels_values=[],
                         not_labels_values=[]):
    
    """
    Getting the index of neurons that its value at the labels is
    in the given list.
    
    Parameters:
    -----------
    neurons: panadas
        all neurons with the labels
    
    attribute_mat: csr_matrixremin
        the sparse matrix of labels and the values
    
    labels_encoder: LabelEncoder
        encoded version of all labels+values
    
    labels_values: list of dic
        each dictionary has a label as a key with assigned with some 
        values. For example: [{'cell_type':['Long projecting','interneuron']},
                {'stain':['lucifer yellow']}]
    
    not_labels_values: list of dic
        each dictionary has a label as a key with assigned with some 
        values. It finds the labels that their values is not
        in the list. For example: [{'cell_type':['Long projecting','interneuron']}]
        finds neurons that are not 'Long projecting' and not 'interneuron'.
        
    Returns:
    --------
    index: numpy
        the indecies of neurons in the dataset
    """    
    n_data = attribute_mat.shape[0]
    index = np.arange(n_data)
    for n in labels_values:
        name = list(n.keys())[0]
        index_in_labels = []
        for label in n[name]:
            index_in_sparse = labels_encoder.transform([name + ' : ' + label])[0]
            having_value = np.where(attribute_mat[:,index_in_sparse].toarray()==1)[0]
            index_in_labels = np.union1d(index_in_labels,
                                         having_value)
        index = np.intersect1d(index, index_in_labels)
    
    for n in not_labels_values:
        name = list(n.keys())[0]
        index_in_labels = []
        for label in n[name]:
            index_in_sparse = labels_encoder.transform([name + ' : ' + label])[0]
            having_value = np.where(attribute_mat[:,index_in_sparse].toarray()==1)[0]
            index_in_labels = np.union1d(index_in_labels,
                                        having_value)
        index_not_in_labels = np.array(list(set(range(n_data)) - set(index_in_labels)))
        index = np.intersect1d(index, index_not_in_labels)
    return index.astype(int)

def get_neurons_with_given_values(neurons,
                                  attribute_mat,
                                  labels_encoder,
                                  labels_values=[],
                                  not_labels_values=[]):
    """
    Getting the dataframe of neurons that its value at the labels is
    in the given list.
    
    Parameters:
    -----------
    neurons: panadas
        all neurons with the labels
    
    attribute_mat: csr_matrixremin
        the sparse matrix of labels and the values
    
    labels_encoder: LabelEncoder
        encoded version of all labels+values
    
    labels_values: list of dic
        each dictionary has a label as a key with assigned with some 
        values. For example: [{'cell_type':['Long projecting','interneuron']},
                {'stain':['lucifer yellow']}]
    
    not_labels_values: list of dic
        each dictionary has a label as a key with assigned with some 
        values. It finds the labels that their values is not
        in the list. For example: [{'cell_type':['Long projecting','interneuron']}]
        finds neurons that are not 'Long projecting' and not 'interneuron'.
        
    Returns:
    --------
    Dataframe of neurons in the dataset
    """ 
    index = get_neurons_in_group(attribute_mat=attribute_mat,
                             labels_encoder=labels_encoder,
                             labels_values=labels_values,
                             not_labels_values=not_labels_values)
    return neurons.iloc[index]



def divide_data_based_label(label, 
                            values,
                            attribute_mat,
                            labels_encoder):
    """
    Accessory for find_comparing_groups 
    """
    groups = []
    lable_in = []
    label_out = []
    for i in range(len(values)+1):
        if i == len(values):
            all_values = []
            for i in values:
                for j in i:
                    all_values.append(j)
            index = get_neurons_in_group(attribute_mat=attribute_mat,
                             labels_encoder=labels_encoder,
                             labels_values=[],
                             not_labels_values=[{label:all_values}]) 
            if len(index)>0:
                
                groups.append(index)
                lable_in.append([])
                label_out.append([{label:all_values}])
        else:
            index = get_neurons_in_group(attribute_mat=attribute_mat,
                             labels_encoder=labels_encoder,
                             labels_values=[{label:values[i]}],
                             not_labels_values=[]) 
            if len(index)>0:
                groups.append(index)
                lable_in.append([{label:values[i]}])
                label_out.append([])
    return groups, lable_in, label_out

def conditioning_on_label(groups_index, 
                          groups_label_in, 
                          groups_lable_out,
                          condition,
                          attribute_mat,
                          labels_encoder):
    """
    Accessory for find_comparing_groups 
    """
    new_groups_index = []
    new_groups_label = []
    new_groups_not_label = []
    for i in range(len(groups_index)):
        ind = groups_index[i]
        groups, label_in, label_out = \
            divide_data_based_label(label=list(condition.keys())[0], 
                    values=list(condition.values())[0],
                    attribute_mat=attribute_mat[ind, :],
                    labels_encoder=labels_encoder)
        for g in range(len(groups)):
            
            new_groups_index.append(ind[groups[g]])
            new_groups_label.append(groups_label_in[i] + label_in[g])
            new_groups_not_label.append(groups_lable_out[i]+label_out[g])

    return new_groups_index, new_groups_label, new_groups_not_label

    
def comparing_groups(regions,
                     conditions,
                     minimums,
                     global_labels,
                     neurons,
                     attribute_mat,
                     labels_encoder):
    """
    Divide the dataset into groups which have the same label and regions.
    Retruning the groups that have at least a minimum number of 
    different values for a given labels (for example the groups
    that have at least two different staining methods).
    
    Parameters:
    -----------
    regions: list
        Example: 
        regions = [
        ['somatosensory', 'primary somatosensory', 'barrel'],
        ['primary visual'],
        ['motor', 'secondary motor', 'primary motor'],
        ['CA1'],
        ['CA3']]
        
    conditions: list
        Example:
        conditions = [
            {'brain_region' : [['layer 1'],
                ['layer 2', 'L2', 'L2i','L2o'],
                ['layer 2-3']]},
            {'brain_region' : [
                ['right'],
                ['left']]},       
            {'gender':[['Male'], 
                ['Female'], 
                ['Male/Female']]},
            {'age_classification': [['adult'],
                            ['young']]}]
                            
    global_labels: list
        Example:
        global_labels = [ {'Physical Integrity':[
             'Dendrites & Axon Complete',
             'Dendrites & Axon Moderate',
             'Processes Moderate']},
     {'experiment_condition':['Control']},
     {'Species Name':['rat']}
                ]
    
    attribute_mat: numpy
        One hat matrix of labels+values
    labels_encoder: LableEncoder
        index of label+values in the attribute_mat
        
    minimums list
        list of all the labels for a given key. 
        example: minimums = [{'stain':[2,10]}, 
                             {'Archive Name': [2,10]}]
  
    Returns:
    --------
    all_groups: list
        list of all indicies in the dataset that lay in the same group
        
    groups_label: list
        The label+values to include for the coorsponding indicies
    
    groups_not_label: list
        The label+values to exclude for the coorsponding indicies
    
    """
    all_groups = []
    groups_label = []
    groups_not_label = []
    for r in range(len(regions)):
        ##### Conditioning on the brain region #####
        labels_values=global_labels+[{'brain_region': regions[r]}]
        n = get_neurons_in_group(attribute_mat=attribute_mat,
                             labels_encoder=labels_encoder,
                             labels_values=labels_values,
                             not_labels_values=[])
        
        ##### Conditioning on the other criterias #####
        if n.shape[0]>0:
            groups_index = [n]
            groups_label_in = [labels_values]
            groups_label_out = [[]]
            for condition in conditions:
                groups_index, groups_label_in, groups_label_out = \
                    conditioning_on_label(groups_index=groups_index, 
                          groups_label_in=groups_label_in, 
                          groups_lable_out=groups_label_out,
                          condition=condition,
                          attribute_mat=attribute_mat,
                          labels_encoder=labels_encoder)  
        
        ##### Minimums requirments #####
        for g in range(len(groups_index)):
            n = neurons.iloc[groups_index[g]]
            label = list(minimums[0].keys())[0]
            min_cat = minimums[0][label][0]
            min_sample = minimums[0][label][1]
            label_values = count_label(neurons=n, label_name=label) 

            if len(label_values)>=min_cat:
                if label_values[min_cat-1][1] > min_sample:
                    list_labels = []
                    for i in range(len(label_values)):
                        if label_values[i][1]>min_sample:
                            list_labels.append(label_values[i][0])
                    label_in = groups_label_in[g]
                    label_in.append({label:list_labels})   
                    
                    g_index = get_neurons_in_group(attribute_mat=attribute_mat,
                                 labels_encoder=labels_encoder,
                                 labels_values=label_in,
                                 not_labels_values=groups_label_out[g])
                    all_groups = all_groups + [g_index]
                    groups_label = groups_label + [label_in]
                    groups_not_label = groups_not_label + [groups_label_out[g]]
    return all_groups, groups_label, groups_not_label

def clean_data(neuron_data):
    """
    This file is to clean the data from neuromorpho.
    It change the key "Average Rall\\'s Ratio" to "Average Rall Ratio"
    and only takes the keys that are relavant to the biology and morphology of
    the neuron. These are the list:
    
    'Physical Integrity',   
    'Species Name', 
    'age_classification', 
    'gender',
    'brain_region',
    'cell_type', 
    'experiment_condition',
    'protocol'
           
    'Max Age', 
    'Min Age'          

    'Morphological Attributes',
    'Structural Domains',
    'Archive Name',
    'original_format',
    'reconstruction_software',
    'reported_value', 
    'slicing_direction',
    'shrinkage_corrected',
    'shrinkage_reported',
    'objective_type',
    'stain',
    'strain'
               
    'magnification', 
    'slicing_thickness',
    'reported_xy',
    'reported_z', 
    'corrected_value', 
    'corrected_xy',      
    'corrected_z'

    'Average Bifurcation Angle Local',
    'Average Bifurcation Angle Remote',
    'Average Contraction',
    'Average Diameter', 
    'Average Rall Ratio',
    'Fractal Dimension',
    'Max Euclidean Distance',
    'Max Path Distance',
    'Max Branch Order',
    'Number of Bifurcations',
    'Number of Branches',
    'Number of Stems', 
    'Overall Depth', 
    'Overall Height', 
    'Overall Width',
    'Partition Asymmetry',
    'Soma Surface',
    'Total Fragmentation',
    'Total Length',
    'Total Surface', 
    'Total Volume', 
    'Soma Surface',
    'volume',
    'soma_surface',
    'surface',
    'Max Weight',
    'Min Weight'
    
    'NeuroMorpho.Org ID',
    'Neuron Name',
    'Primary Article Reference',
    'Related Article Reference',
    'Secondary Article Reference',
    '_links',
    'deposition_date',
    'link in neuromorpho',
    'neuron_id', 
    'neuron_name',
    'note',
    'png_url', 
    'reference_doi',
    'reference_pmid',
    'scientific_name',
    'upload_date'
    
    'swc'
    
    Parameters:
    -----------
    
    Returns:
    --------
    
    """
    neuron_data["Average Rall Ratio"] = neuron_data["Average Rall\\'s Ratio"]
    bio_cat = ['Physical Integrity',   
               'Species Name', 
               'age_classification', 
               'gender',
               'brain_region',
               'cell_type', 
               'experiment_condition',
               'protocol']

    bio_quant = ['Max Age', 
                 'Min Age']

    non_bio_cat = ['Morphological Attributes',
                   'Structural Domains',
                   'Archive Name',
                   'original_format',
                   'reconstruction_software',
                   'reported_value', 
                   'slicing_direction',
                   'shrinkage_corrected',
                   'shrinkage_reported',
                   'objective_type',
                   'stain',
                   'strain']

    non_bio_quant = ['magnification', 
                     'slicing_thickness',
                     'reported_xy',
                     'reported_z', 
                     'corrected_value', 
                     'corrected_xy',      
                     'corrected_z']

    morphometry = ['Average Bifurcation Angle Local',
                   'Average Bifurcation Angle Remote',
                   'Average Contraction',
                   'Average Diameter', 
                   "Average Rall\\'s Ratio",
                   'Fractal Dimension',
                   'Max Euclidean Distance',
                   'Max Path Distance',
                   'Max Branch Order',
                   'Number of Bifurcations',
                   'Number of Branches',
                   'Number of Stems', 
                   'Overall Depth', 
                   'Overall Height', 
                   'Overall Width',
                   'Partition Asymmetry',
                   'Soma Surface',
                   'Total Fragmentation',
                   'Total Length',
                   'Total Surface', 
                   'Total Volume', 
                   'Soma Surface',
                   'volume',
                   'soma_surface',
                   'surface',
                   'Max Weight',
                   'Min Weight']

    irrelavant = ['NeuroMorpho.Org ID',
                  'Neuron Name',
                  'Primary Article Reference',
                  'Related Article Reference',
                  'Secondary Article Reference',
                  '_links',
                  'deposition_date',
                  'link in neuromorpho',
                  'neuron_id', 
                  'neuron_name',
                  'note',
                  'png_url', 
                  'reference_doi',
                  'reference_pmid',
                  'scientific_name',
                  'upload_date']
    swc = ['swc']
    all_labels = bio_cat + bio_quant + \
        non_bio_cat + non_bio_quant + morphometry + irrelavant+swc

    new_neuron_data = neuron_data[all_labels]
    new_neuron_data["Average Rall Ratio"] = new_neuron_data["Average Rall\\'s Ratio"]
    return new_neuron_data

def selection_util(neurons, labels):
    List_all_labels = get_all_labels(neurons=neurons,
                                labels = labels)
    labels_encoder = preprocessing.LabelEncoder()
    labels_encoder.fit(List_all_labels)

    size = neurons.shape[0]
    attribute_mat = csr_matrix((size, len(labels_encoder.classes_)))
    for i in range(size):
        if np.mod(i,100)==0:
            print(i)
        for name in labels:
            label = neurons[name][i]
            if isinstance(label, str):
                index = labels_encoder.transform([name + ' : ' + label])[0]
                attribute_mat[i, index]= 1
            if isinstance(label, list):
                for lab in label:
                    index = labels_encoder.transform([name + ' : ' + lab])[0]
                    attribute_mat[i, index]= 1
    return labels_encoder, attribute_mat


# def feature_ext(morph_data_label, str_to_float=0):
#     """Turning each morphological feature in the neuromorpho to
#     a number and returning the array of the features.
    
#     Parameters:
#     -----------
#     morph_data_label: list
#         list of value of the features for different neurons.
#         for example it can be: morph_data['Max Path Distance']
    
#     str_to_float: int
#         the value of the label is only considered to -(str_to_float)
#         for example if morph_data['Max Path Distance']='258.27 mm'
#         then str_to_float=-3 only takes '258.27'
    
#     feature_label: numpy array
#         array of the features.
        
#     """
#     n_morp_daya = morph_data_label.shape[0]
#     feature_label = np.zeros(n_morp_daya)
#     for i in range(n_morp_daya):
#         if isinstance(morph_data_label[i], float):
#             feature_label[i] = morph_data_label[i]
#         else:
#             if str_to_float==0:
#                 feature_label[i] = float(morph_data_label[i])
#             else:
#                 feature_label[i] = float(morph_data_label[i][:str_to_float])
#     return feature_label

# def feature_matrix_from_neuromorpho(morph_data):
#     """Returning the feature matrix.
    
#     Parameteres:
#     ------------
#     morph_data: Dataframe
#         a dataframe (panadas) that each row is a neuron that is
#         registered in the neuromorpho.org and the columns are the 
#         attributes of neuron (swc, Archive Name,...).
    
#     Returns:
#     --------
#     feature_matrix: numpy array
#         a 2D matrix that rows are neurons and columns are:
        
#         0 = ave_bif_ang_local

#         1 = ave_bif_ang_remote

#         2 = ave_cont

#         3 = ave_diam

#         4 = ave_rall

#         5 = frac_diam

#         6 = max_branch

#         7 = max_euc_dis

#         8 = max_path_dis

#         9 = max_weigth

#         10 = n_bif

#         11 = n_branch

#         12 = n_stem

#         13 = overal_depth

#         14 = overal_heigth

#         15 = overal_width

#         16 = p_asym

#         17 = soma_surf

#         18 = tot_len

#         19 = tot_surf

#         20 = tot_vol

#         21 = tot_frag 
#     """
#     ave_bif_ang_local = feature_ext(morph_data['Average Bifurcation Angle Local'],
#                                     str_to_float=-3)
#     ave_bif_ang_remote = feature_ext(morph_data['Average Bifurcation Angle Remote'],
#                                     str_to_float=-3)
#     ave_cont = feature_ext(morph_data['Average Contraction'],
#                           str_to_float=0)
#     ave_diam = feature_ext(morph_data['Average Diameter'],
#                           str_to_float=-3)
#     ave_rall = feature_ext(morph_data["Average Rall's Ratio"],
#                           str_to_float=0)
#     frac_diam = feature_ext(morph_data['Fractal Dimension'],
#                           str_to_float=0)
#     max_branch = feature_ext(morph_data['Max Branch Order'],
#                           str_to_float=0)
#     max_euc_dis = feature_ext(morph_data['Max Euclidean Distance'],
#                           str_to_float=-3)
  
#     max_path_dis = feature_ext(morph_data['Max Path Distance'],
#                           str_to_float=-3)

#     a = deepcopy(morph_data['Max Weight'])
#     a[a=='Not reported'] = '0 grams'
#     max_weigth = feature_ext(a,
#                           str_to_float=-5)
#     n_bif = feature_ext(morph_data['Number of Bifurcations'],
#                                     str_to_float=0)
#     n_branch = feature_ext(morph_data['Number of Branches'],
#                                     str_to_float=0)
#     a = morph_data['Number of Stems']
#     a[a=='N/A'] = np.nan
#     n_stem = feature_ext(a,str_to_float=0)
#     overal_depth = feature_ext(morph_data['Overall Depth'], str_to_float=-3)
#     overal_heigth = feature_ext(morph_data['Overall Height'], str_to_float=-3)

#     overal_width = feature_ext(morph_data['Overall Width'], str_to_float=-3)

#     p_asym = feature_ext(morph_data['Partition Asymmetry'], str_to_float=0)
#     a = morph_data['Soma Surface']
#     a[a=='N/A'] = np.nan
#     a[a=='Soma Surface'] = np.nan
#     soma_surf = feature_ext(a, str_to_float=-4)
#     tot_len = feature_ext(morph_data['Total Length'], str_to_float=-3)

#     tot_surf = feature_ext(morph_data['Total Surface'], str_to_float=-4)

#     tot_vol = feature_ext(morph_data['Total Volume'], str_to_float=-4)
#     tot_frag = feature_ext(morph_data['Total Fragmentation'], str_to_float=0)

#     feature_matrix = np.zeros([morph_data.shape[0], 22])
#     feature_matrix[:,0] = ave_bif_ang_local
#     feature_matrix[:,1] = ave_bif_ang_remote
#     feature_matrix[:,2] = ave_cont
#     feature_matrix[:,3] = ave_diam
#     feature_matrix[:,4] = ave_rall
#     feature_matrix[:,5] = frac_diam
#     feature_matrix[:,6] = max_branch
#     feature_matrix[:,7] = max_euc_dis
#     feature_matrix[:,8] = max_path_dis
#     feature_matrix[:,9] = max_weigth
#     feature_matrix[:,10] = n_bif
#     feature_matrix[:,11] = n_branch
#     feature_matrix[:,12] = n_stem
#     feature_matrix[:,13] = overal_depth
#     feature_matrix[:,14] = overal_heigth
#     feature_matrix[:,15] = overal_width
#     feature_matrix[:,16] = p_asym
#     feature_matrix[:,17] = soma_surf
#     feature_matrix[:,18] = tot_len
#     feature_matrix[:,19] = tot_surf
#     feature_matrix[:,20] = tot_vol
#     feature_matrix[:,21] = tot_frag

#     feature_matrix[np.isnan(feature_matrix)] = 0
#     if morph_data.shape[0] > 75000:
#         max_euc_dis[77536] = 4000
#         max_path_dis[77536] = 4000
#         overal_heigth[77536] = 50000
#         overal_width[77536] = 10000
#         tot_len[77536] = 80000    
#         tot_surf[77536] = 200000   
#         tot_vol[77536] = 20000
#     return feature_matrix
