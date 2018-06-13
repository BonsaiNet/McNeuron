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
from bs4 import BeautifulSoup
from unidecode import unidecode
from copy import deepcopy

def get_dict_from_table(trs):
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

def swc(swc_html):
    neuron = np.zeros([1,7])
    for line in swc_html:
        line = line.lstrip()
        #print line
        if line[0] is not '#' and len(line) > 2:
            #print line
            l= '['+re.sub('(\d) +', r'\1,', line[:-1])+']'
            #print l
            l= re.sub('(\.) ', r'\1,', l)
            #print l
            neuron = np.append(neuron, np.expand_dims(np.array(ast.literal_eval(l)),axis=0),axis=0)
    return neuron[1:,:]

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
    # Neurmorpho 7.2 
            #txt = urllib.urlopen("http://cng.gmu.edu:8080/solr/nmo/query?q=collector:(%22nmo_"+format(nmo, '05d')+"%22%20)&start=0&rows=500").read()      
    #         if neuron_dict['response']['numFound'] == 1:    
    #             neuron_name = neuron_dict['response']['docs'][0]['neuron_name']
    #             archive_name1 = neuron_dict['response']['docs'][0]['archive']
    #             archive_name = re.sub(' ', '%20',archive_name1)
    #             neuromorpho_link = 'http://neuromorpho.org/neuron_info.jsp?neuron_name='+ neuron_name
    #             neuromorpho_link_swc = 'http://neuromorpho.org/dableFiles/' + archive_name.lower() + '/CNG%20version/' + neuron_name + '.CNG.swc'
    #             metadata_html = urllib.urlopen(neuromorpho_link).read()
    #             response = urllib.urlopen(neuromorpho_link_swc)
    #             if response.getcode() != 404:
    #                 swc_html = response.readlines()
    #                 dic_all = get_metadata(metadata_html)
    #                 dic_all['link in neuromorpho'] = str(neuromorpho_link)

    #                 dic_all['swc'] = swc(swc_html)
    #                 all_neuron.append(dic_all)

    # Neurmorpho 7.3 
        try:
            if np.mod(nmo,1)==0:
                print(nmo)


            txt = urllib.urlopen("http://cng.gmu.edu:8080/search/keyword/summary?q=nmo_"+format(nmo, '05d')).read()
            neuron_dict = json.loads(txt.decode("utf-8"))
            if len(neuron_dict) != 0:
                neuron_dict = neuron_dict[0]
                neuron_name = neuron_dict['neuron_name']
                archive_name1 = neuron_dict['archive']
                archive_name = re.sub(' ', '%20',archive_name1)
                neuromorpho_link = 'http://neuromorpho.org/neuron_info.jsp?neuron_name='+ neuron_name
                neuromorpho_link_swc = 'http://neuromorpho.org/dableFiles/' + archive_name.lower() + '/CNG%20version/' + neuron_name + '.CNG.swc'
                metadata_html = urllib.urlopen(neuromorpho_link).read()
                response = urllib.urlopen(neuromorpho_link_swc)
                if response.getcode() != 404:
                    swc_html = response.readlines()
                    metadata_html = metadata_html.replace('</tr>\n</tr>','</tr>',1)
                    dic_all = get_metadata(metadata_html)
                    dic_all['link in neuromorpho'] = str(neuromorpho_link)

                    dic_all['swc'] = swc(swc_html)
                    all_neuron.append(dic_all)
        except:
            errors.append(nmo)
        morph_data = pd.DataFrame(all_neuron)
    return morph_data, errors

def read_data(meta_data_path, swc_path):
    morph_data = pd.read_pickle(meta_data_path)
    morph_data['swc'] = regular_swc = np.load(swc_path)
    return morph_data


def feature_ext(morph_data_label, str_to_float=0):
    """Turning each morphological feature in the neuromorpho to
    a number and returning the array of the features.
    
    Parameters:
    -----------
    morph_data_label: list
        list of value of the features for different neurons.
        for example it can be: morph_data['Max Path Distance']
    
    str_to_float: int
        the value of the label is only considered to -(str_to_float)
        for example if morph_data['Max Path Distance']='258.27 mm'
        then str_to_float=-3 only takes '258.27'
    
    feature_label: numpy array
        array of the features.
        
    """
    n_morp_daya = morph_data_label.shape[0]
    feature_label = np.zeros(n_morp_daya)
    for i in range(n_morp_daya):
        if isinstance(morph_data_label[i], float):
            feature_label[i] = morph_data_label[i]
        else:
            if str_to_float==0:
                feature_label[i] = float(morph_data_label[i])
            else:
                feature_label[i] = float(morph_data_label[i][:str_to_float])
    return feature_label

def feature_matrix_from_neuromorpho(morph_data):
    """Returning the feature matrix.
    
    Parameteres:
    ------------
    morph_data: Dataframe
        a dataframe (panadas) that each row is a neuron that is
        registered in the neuromorpho.org and the columns are the 
        attributes of neuron (swc, Archive Name,...).
    
    Returns:
    --------
    feature_matrix: numpy array
        a 2D matrix that rows are neurons and columns are:
        
        0 = ave_bif_ang_local

        1 = ave_bif_ang_remote

        2 = ave_cont

        3 = ave_diam

        4 = ave_rall

        5 = frac_diam

        6 = max_branch

        7 = max_euc_dis

        8 = max_path_dis

        9 = max_weigth

        10 = n_bif

        11 = n_branch

        12 = n_stem

        13 = overal_depth

        14 = overal_heigth

        15 = overal_width

        16 = p_asym

        17 = soma_surf

        18 = tot_len

        19 = tot_surf

        20 = tot_vol

        21 = tot_frag 
    """
    ave_bif_ang_local = feature_ext(morph_data['Average Bifurcation Angle Local'],
                                    str_to_float=-3)
    ave_bif_ang_remote = feature_ext(morph_data['Average Bifurcation Angle Remote'],
                                    str_to_float=-3)
    ave_cont = feature_ext(morph_data['Average Contraction'],
                          str_to_float=0)
    ave_diam = feature_ext(morph_data['Average Diameter'],
                          str_to_float=-3)
    ave_rall = feature_ext(morph_data["Average Rall's Ratio"],
                          str_to_float=0)
    frac_diam = feature_ext(morph_data['Fractal Dimension'],
                          str_to_float=0)
    max_branch = feature_ext(morph_data['Max Branch Order'],
                          str_to_float=0)
    max_euc_dis = feature_ext(morph_data['Max Euclidean Distance'],
                          str_to_float=-3)
  
    max_path_dis = feature_ext(morph_data['Max Path Distance'],
                          str_to_float=-3)

    a = deepcopy(morph_data['Max Weight'])
    a[a=='Not reported'] = '0 grams'
    max_weigth = feature_ext(a,
                          str_to_float=-5)
    n_bif = feature_ext(morph_data['Number of Bifurcations'],
                                    str_to_float=0)
    n_branch = feature_ext(morph_data['Number of Branches'],
                                    str_to_float=0)
    a = morph_data['Number of Stems']
    a[a=='N/A'] = np.nan
    n_stem = feature_ext(a,str_to_float=0)
    overal_depth = feature_ext(morph_data['Overall Depth'], str_to_float=-3)
    overal_heigth = feature_ext(morph_data['Overall Height'], str_to_float=-3)

    overal_width = feature_ext(morph_data['Overall Width'], str_to_float=-3)

    p_asym = feature_ext(morph_data['Partition Asymmetry'], str_to_float=0)
    a = morph_data['Soma Surface']
    a[a=='N/A'] = np.nan
    a[a=='Soma Surface'] = np.nan
    soma_surf = feature_ext(a, str_to_float=-4)
    tot_len = feature_ext(morph_data['Total Length'], str_to_float=-3)

    tot_surf = feature_ext(morph_data['Total Surface'], str_to_float=-4)

    tot_vol = feature_ext(morph_data['Total Volume'], str_to_float=-4)
    tot_frag = feature_ext(morph_data['Total Fragmentation'], str_to_float=0)

    feature_matrix = np.zeros([morph_data.shape[0], 22])
    feature_matrix[:,0] = ave_bif_ang_local
    feature_matrix[:,1] = ave_bif_ang_remote
    feature_matrix[:,2] = ave_cont
    feature_matrix[:,3] = ave_diam
    feature_matrix[:,4] = ave_rall
    feature_matrix[:,5] = frac_diam
    feature_matrix[:,6] = max_branch
    feature_matrix[:,7] = max_euc_dis
    feature_matrix[:,8] = max_path_dis
    feature_matrix[:,9] = max_weigth
    feature_matrix[:,10] = n_bif
    feature_matrix[:,11] = n_branch
    feature_matrix[:,12] = n_stem
    feature_matrix[:,13] = overal_depth
    feature_matrix[:,14] = overal_heigth
    feature_matrix[:,15] = overal_width
    feature_matrix[:,16] = p_asym
    feature_matrix[:,17] = soma_surf
    feature_matrix[:,18] = tot_len
    feature_matrix[:,19] = tot_surf
    feature_matrix[:,20] = tot_vol
    feature_matrix[:,21] = tot_frag

    feature_matrix[np.isnan(feature_matrix)] = 0
    if morph_data.shape[0] > 75000:
        max_euc_dis[77536] = 4000
        max_path_dis[77536] = 4000
        overal_heigth[77536] = 50000
        overal_width[77536] = 10000
        tot_len[77536] = 80000    
        tot_surf[77536] = 200000   
        tot_vol[77536] = 20000
    return feature_matrix