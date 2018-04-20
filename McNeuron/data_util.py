Jupyter Notebook data_util.py 02/18/2018
Python
File
Edit
View
Language

"""A set of utilities for downloading and reading neuromorpho data"""
import sys
import urllib
import ast
import numpy as np
import pandas as pd
import pickle
import re
import json
from urllib2 import Request, urlopen
from bs4 import BeautifulSoup
from unidecode import unidecode
​
def get_dict_from_table(trs):
    table_dict = {}
    for tr in trs:
        if len(tr.find_all('td')) == 2:
            k, v = [t.text for t in tr.find_all('td')]
            k = unidecode(k.replace(':', '').strip())
            v = unidecode(v.replace(':', '').strip())
            table_dict[k] = v
    return table_dict
​
​
def find_archive_link(a):
    for a in soup.find_all('a'):
        if a is not None and a.find('input') is not None:
            if 'Link to original archive' == a.find('input').attrs['value']:
                archive_link = a.attrs['href']
    return a
​
def get_metadata(metadata_html):
    soup = BeautifulSoup(metadata_html, 'html.parser')
    table1 = soup.find_all('tbody')[1]
    trs = table1.find_all('tr')[2].find_all('tr')
    d1 = get_dict_from_table(trs)
​

    table2 = soup.find_all('tbody')[4]
    trs = table2.find_all('tr')
    d2 = get_dict_from_table(trs)
​
    table3 = soup.find_all('tbody')[8]
    trs = table3.find_all('tr')
    d3 = get_dict_from_table(trs)

    dic = {}
    dic.update(d1)
    dic.update(d2)
    dic.update(d3)
    return dic
​
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
​
def download_neuromorpho(start_nmo=1, end_nmo=50):
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
​
    #                 dic_all['swc'] = swc(swc_html)
    #                 all_neuron.append(dic_all)
​
    # Neurmorpho 7.3
        try:
            if np.mod(nmo,1)==0:
                print(nmo)
​
​
            txt = urllib.urlopen("http://cng.gmu.edu:8080/search/keyword/summary?q=nmo_"+format(nmo, '05d')).read()
            neuron_dict = json.loads(txt.decode("utf-8"))
            if len(neuron_dict) != 0:
                neuron_dict = neuron_dict[0]
                neuron_name = neuron_dict['neuron_name']
                archive_name1 = neuron_dict['archive']
                archive_name = re.sub(' ', '%20',archive_name1)
