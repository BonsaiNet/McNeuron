"""Asymmetric"""

from copy import deepcopy
import numpy as np
from collections import Counter
import pickle
from numpy import sqrt as sqrt
import McNeuron
from McNeuron import Neuron
from scipy.optimize import fmin
import math
from McNeuron import subsample
from McNeuron import visualize
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn import preprocessing
from decimal import Decimal, getcontext
from scipy.sparse import csr_matrix
import matplotlib.patches as mpatches
getcontext().prec = 50

def extract_stat(data, length=1, part='all', show_count=False, count_hit=1000):
    n_neuron = len(data)
    BBlist = np.zeros(n_neuron)
    BDlist = np.zeros(n_neuron)
    DDlist = np.zeros(n_neuron)
    Colist = np.zeros(n_neuron)
    nBlist = np.zeros(n_neuron)
    Ilist = np.zeros(n_neuron)
    nonIlist = np.zeros(n_neuron)
    Blist = []
    Clist = []
    Dlist = []
    Breglist = []
    Dreglist = []
    BBdepthlist = []
    BDdepthlist = []
    DDdepthlist = []
    BBdepthreglist = []
    BDdepthreglist = []
    DDdepthreglist = []
    for index in range(n_neuron):
        if np.mod(index,count_hit) == 0:
            if show_count:
                print(index)
        swc_matrix = data[index]
        BB, BD, DD, Co, B, C, D, nB, I, nonI,\
            BBdepth, BDdepth, DDdepth, Breg, Dreg , BBdepthreg, \
            BDdepthreg, DDdepthreg = get_stat_for_single_neuron(swc_matrix, length, part)

        BBlist[index] = BB
        BDlist[index] = BD
        DDlist[index] = DD
        Colist[index] = Co
        Blist.append(B)
        Clist.append(C)
        Dlist.append(D) 
        nBlist[index] = nB
        Ilist[index] = I
        nonIlist[index] = nonI
        BBdepthlist.append(BBdepth)
        BDdepthlist.append(BDdepth)
        DDdepthlist.append(DDdepth)
        Breglist.append(Breg)
        Dreglist.append(Dreg)
        BBdepthreglist.append(BBdepthreg)
        BDdepthreglist.append(BDdepthreg)
        DDdepthreglist.append(DDdepthreg)
    return BBlist, BDlist, DDlist, Colist, Blist, Clist, Dlist, nBlist, Ilist, nonIlist, BBdepthlist, BDdepthlist, DDdepthlist, Breglist, Dreglist, BBdepthreglist, BDdepthreglist, DDdepthreglist


def get_stat(database):
    """
    Get Statistics of database
    """
    BBlist = 0
    BDlist = 0
    DDlist = 0
    Colist = 0
    nBlist = 0
    Ilist = 0
    nonIlist = 0
    startBlist = 0
    startClist = 0
    
    Blist, Clist, Dlist = np.array([]), np.array([]), np.array([])
    Breglist, Dreglist = np.array([]), np.array([])
    BBdepthlist, BDdepthlist, DDdepthlist = np.array([]), np.array([]), np.array([])
    BBdepthreglist, BDdepthreglist, DDdepthreglist = np.array([]), np.array([]), np.array([])

    for swc in database:
        BB, BD, DD, Co, B, C, D, nB, I, nonI, BBdepth, BDdepth, DDdepth, Breg, Dreg, BBdepthreg, BDdepthreg, DDdepthreg = get_stat_for_single_neuron(swc)

        nBlist += nB
        Ilist += I
        Colist += Co
        nonIlist += nonI
        BBlist += BB
        BDlist += BD
        DDlist += DD
        startBlist += B[1]
        startClist += C[1]
        Blist = sum_(Blist, B[1:])
        Clist = sum_(Clist, C[1:])
        Dlist = sum_(Dlist, D[1:])

        Breglist = sum_(Breglist, Breg[2:])
        Dreglist = sum_(Dreglist, Dreg[2:])

        BBdepthlist = sum_(BBdepthlist , BBdepth[1:-1])
        BDdepthlist = sum_(BDdepthlist , BDdepth[1:-1])
        DDdepthlist = sum_(DDdepthlist , DDdepth[1:-1])

        BBdepthreglist = sum_(BBdepthreglist , BBdepthreg[1:-1])
        BDdepthreglist = sum_(BDdepthreglist , BDdepthreg[1:-1])
        DDdepthreglist = sum_(DDdepthreglist , DDdepthreg[1:-1])

    return BBlist, BDlist, DDlist, Colist, Blist, Clist, Dlist, nBlist, Ilist, nonIlist, BBdepthlist, BDdepthlist, DDdepthlist, Breglist, Dreglist , BBdepthreglist, BDdepthreglist, DDdepthreglist, startBlist, startClist

def subset_stat(stat, index='all'):
    """
    Get Statistics of database
    """
    BB, BD, DD, Co, B, C, D, nB, I, nonI, BBdepth, BDdepth, DDdepth, Breg, Dreg, BBdepthreg, BDdepthreg, DDdepthreg = stat
    if index is 'all':
        index = np.arange(len(BB))
    BBlist = 0
    BDlist = 0
    DDlist = 0
    Colist = 0
    nBlist = 0
    Ilist = 0
    nonIlist = 0
    startBlist = 0
    startClist = 0
    
    Blist, Clist, Dlist = np.array([]), np.array([]), np.array([])
    Breglist, Dreglist = np.array([]), np.array([])
    BBdepthlist, BDdepthlist, DDdepthlist = np.array([]), np.array([]), np.array([])
    BBdepthreglist, BDdepthreglist, DDdepthreglist = np.array([]), np.array([]), np.array([])

    for i in index:

        nBlist += nB[i]
        Ilist += I[i]
        Colist += Co[i]
        nonIlist += nonI[i]
        BBlist += BB[i]
        BDlist += BD[i]
        DDlist += DD[i]
        startBlist += B[i][1]
        startClist += C[i][1]
        Blist = sum_(Blist, B[i][1:])
        Clist = sum_(Clist, C[i][1:])
        Dlist = sum_(Dlist, D[i][1:])

        Breglist = sum_(Breglist, Breg[i][2:])
        Dreglist = sum_(Dreglist, Dreg[i][2:])


        BBdepthlist = sum_(BBdepthlist , BBdepth[i][1:-1])
        BDdepthlist = sum_(BDdepthlist , BDdepth[i][1:-1])
        DDdepthlist = sum_(DDdepthlist , DDdepth[i][1:-1])

        BBdepthreglist = sum_(BBdepthreglist , BBdepthreg[i][1:-1])
        BDdepthreglist = sum_(BDdepthreglist , BDdepthreg[i][1:-1])
        DDdepthreglist = sum_(DDdepthreglist , DDdepthreg[i][1:-1])

    
    return BBlist, BDlist, DDlist, Colist, Blist, Clist, Dlist, nBlist, Ilist, nonIlist, BBdepthlist, BDdepthlist, DDdepthlist, Breglist, Dreglist , BBdepthreglist, BDdepthreglist, DDdepthreglist, startBlist, startClist

def fit(stat):
    BBlist, BDlist, DDlist, Colist, Blist, Clist, Dlist, nBlist, Ilist, nonIlist, BBdepthlist, BDdepthlist, DDdepthlist, Breglist, Dreglist , BBdepthreglist, BDdepthreglist, DDdepthreglist, startBlist, startClist = stat

    sy_nd_nx_p = ML_sym(nBlist, Ilist)
    
    as_nd_nx_pa, as_nd_nx_pb = ML_asym(BBlist, BDlist, DDlist)

    sy_de_nx_pb, sy_de_nx_pd = ML_sym_depth(Breglist, Dreglist)

    as_de_nx_pa, as_de_nx_pb = ML_asym_depth(BBdepthreglist, BDdepthreglist, DDdepthreglist)
    
    sy_nd_co_pb, sy_nd_co_pc = ML_sym_with_C(nBlist, Colist, Ilist, nonIlist, startBlist, startClist)
    
    as_nd_co_pa, as_nd_co_pb, as_nd_co_pc = ML_asym_with_C(BBlist, BDlist, DDlist, Colist, startBlist, startClist)
    
    sy_de_co_pb, sy_de_co_pc, sy_de_co_pd = ML_sym_depth_with_C(Blist, Clist, Dlist)
    
    as_de_co_pa, as_de_co_pb, as_de_co_pc = ML_asym_depth_with_C(BBdepthlist, BDdepthlist, DDdepthlist, Clist, startBlist, startClist)
    
    return sy_nd_nx_p, as_nd_nx_pa, as_nd_nx_pb, sy_de_nx_pb, sy_de_nx_pd   ,as_de_nx_pa,    as_de_nx_pb,    sy_nd_co_pb,    sy_nd_co_pc,    as_nd_co_pa,    as_nd_co_pb,    as_nd_co_pc,    sy_de_co_pb,    sy_de_co_pc,    sy_de_co_pd,    as_de_co_pa,    as_de_co_pb,    as_de_co_pc

def score(para, stat):
    
    BBlist, BDlist, DDlist, Colist, Blist, Clist, Dlist, nBlist, Ilist, nonIlist, BBdepthlist, BDdepthlist, DDdepthlist, Breglist, Dreglist , BBdepthreglist, BDdepthreglist, DDdepthreglist, startBlist, startClist = stat

    sy_nd_nx_p, as_nd_nx_pa, as_nd_nx_pb, sy_de_nx_pb, sy_de_nx_pd   ,as_de_nx_pa,    as_de_nx_pb,    sy_nd_co_pb,    sy_nd_co_pc,    as_nd_co_pa,    as_nd_co_pb,    as_nd_co_pc,    sy_de_co_pb,    sy_de_co_pc,    sy_de_co_pd,    as_de_co_pa,    as_de_co_pb,    as_de_co_pc = para
    
    sy_nd_nx_l = likelihood_sym(sy_nd_nx_p, nBlist, Ilist)

    as_nd_nx_l = likelihood_asym_simple([as_nd_nx_pa, as_nd_nx_pb], BBlist, BDlist, DDlist)
    
    even_len = min(len(sy_de_nx_pb),len(Breglist))
    sy_de_nx_l = likelihood_sym_depth([sy_de_nx_pb[0:even_len], sy_de_nx_pd[0:even_len]], Breglist[0:even_len], Dreglist[0:even_len])
    
    even_len = min(len(as_de_nx_pa),len(BBdepthreglist))
    as_de_nx_l = likelihood_asym_depth([as_de_nx_pa[0:even_len], as_de_nx_pb[0:even_len]], BBdepthreglist[0:even_len], BDdepthreglist[0:even_len], DDdepthreglist[0:even_len])
    
    sy_nd_co_l = likelihood_sym_with_C([sy_nd_co_pb, sy_nd_co_pc], nBlist, Colist, Ilist, nonIlist, startBlist, startClist)

    as_nd_co_l = likelihood_asym_with_C([as_nd_co_pa, as_nd_co_pb, as_nd_co_pc], BBlist, BDlist, DDlist, Colist, startBlist, startClist)
    
    even_len = min(len(sy_de_co_pb),len(Blist))
    sy_de_co_l = likelihood_sym_depth_with_C([sy_de_co_pb[0:even_len], sy_de_co_pc[0:even_len], sy_de_co_pd[0:even_len] ], Blist[0:even_len], Clist[0:even_len], Dlist[0:even_len])
    
    even_len = min(len(as_de_co_pa),len(BBdepthlist))
    as_de_co_l = likelihood_asym_depth_with_C([as_de_co_pa[0:even_len], as_de_co_pb[0:even_len], as_de_co_pc[0:even_len]], BBdepthlist[0:even_len], BDdepthlist[0:even_len], DDdepthlist[0:even_len], Clist[0:even_len], startBlist, startClist)
    
    return sy_nd_nx_l, as_nd_nx_l, sy_de_nx_l, as_de_nx_l, sy_nd_co_l, as_nd_co_l, sy_de_co_l, as_de_co_l


def show_fit(para, length_stat_vec='inf', save=False, save_path=''):
    sy_nd_nx_p, as_nd_nx_pa, as_nd_nx_pb, sy_de_nx_pb, sy_de_nx_pd   ,as_de_nx_pa,  \
    as_de_nx_pb,    sy_nd_co_pb,    sy_nd_co_pc,    as_nd_co_pa,    as_nd_co_pb,  \
    as_nd_co_pc,    sy_de_co_pb,    sy_de_co_pc,    sy_de_co_pd,    as_de_co_pa,  \
    as_de_co_pb,    as_de_co_pc = para
    data = {}
    data["branching probability of Symmetric Depth-dependent without Extension"] = sy_de_nx_pb
    data["A probability of Asymmetric Depth-dependent without Extension"] = as_de_nx_pa
    data["B probability of Asymmetric Depth-dependent without Extension"] = as_de_nx_pb
    data["branching prob of Symmetric Depth-dependent with Extension"] = sy_de_co_pb
    data["extension prob of Symmetric Depth-dependent with Extension"] = sy_de_co_pc
    data["A probability of Asymmetric Depth-dependent with Extension"] = as_de_co_pa
    data["B probability of Asymmetric Depth-dependent with Extension"] = as_de_co_pb
    data["extension probability of Asymmetric Depth-dependent with Extension"] = as_de_co_pc
    data['sym_no-ext'] = sy_nd_nx_p
    data['pa_asym_no-ext'] = as_nd_nx_pa 
    data['pb_asym_no-ext'] = as_nd_nx_pb
    data['pb_sym_ext'] = sy_nd_co_pb
    data['pc_sym_ext'] = sy_nd_co_pc
    data['pa_asym_ext'] = as_nd_co_pa
    data['pb_asym_ext'] = as_nd_co_pb 
    data['pc_asym_ext'] = as_nd_co_pc 
    
    fig = plt.figure(figsize=(15, 8)) 
    gs = gridspec.GridSpec(2, 3) 
    
    ax0 = plt.subplot(gs[0])
    pa, = ax0.plot(cut_vec(data["branching probability of Symmetric Depth-dependent without Extension"], length_stat_vec),label="branching probability")
    ax0.set_title("Symmetric Depth-dependent without Extension")
    ax0.legend(handles=[pa])
    
    ax1 = plt.subplot(gs[1])
    pa, = ax1.plot(cut_vec(data["A probability of Asymmetric Depth-dependent without Extension"], length_stat_vec), label="A probability")
    pb, = ax1.plot(cut_vec(data["B probability of Asymmetric Depth-dependent without Extension"], length_stat_vec), label="B probability")
    ax1.legend(handles=[pa, pb])
    ax1.set_title("Asymmetric Depth-dependent without Extension")

    ax2 = plt.subplot(gs[2])        
    pa, = ax2.plot(cut_vec(data["branching prob of Symmetric Depth-dependent with Extension"], length_stat_vec), label="branching prob")
    pb, = ax2.plot(cut_vec(data["extension prob of Symmetric Depth-dependent with Extension"], length_stat_vec), label="extension prob")
    ax2.legend(handles=[pa, pb])
    ax2.set_title("Symmetric Depth-dependent with Extension")

    ax3 = plt.subplot(gs[3])     
    pa, = ax3.plot(cut_vec(data["A probability of Asymmetric Depth-dependent with Extension"], length_stat_vec), label="A probability")
    pb, = ax3.plot(cut_vec(data["B probability of Asymmetric Depth-dependent with Extension"], length_stat_vec), label="B probability")
    pc, = ax3.plot(cut_vec(data["extension probability of Asymmetric Depth-dependent with Extension"], length_stat_vec), label="extension probability")
    ax3.legend(handles=[pa, pb, pc])
    ax3.set_title("Asymmetric Depth-dependent with Extension")
 
    ax4 = plt.subplot(gs[4])  

    width=.35
    ind = np.arange(8)
    rects1 = ax4.bar(ind,
                    np.array([sy_nd_nx_p, 
                              as_nd_nx_pa, 
                              as_nd_nx_pb,
                              sy_nd_co_pb, 
                              sy_nd_co_pc,
                              as_nd_co_pa, 
                              as_nd_co_pb,
                              as_nd_co_pc]), 
                    width=width, 
                    color='r')
    ax4.set_ylabel('p')
    ax4.set_title('fitted probabilities')
    ax4.set_xticks(ind)
    ax4.set_xticklabels(('sym_no-ext',
                        'pa_asym_no-ext',
                        'pb_asym_no-ext', 
                        'pb_sym_ext', 
                        'pc_sym_ext',
                        'pa_asym_ext',
                        'pb_asym_ext', 
                        'pc_asym_ext'), rotation='vertical')


    if save:
        plt.savefig(save_path+"fitted probabilities.eps")
        
    plt.show()
    return data
def cut_vec(v, length):
    if length is 'inf':
        return v
    elif(len(v)>length):
        return v[0:length]
    else:
        return v
    
def show_stat(stat, length_stat_vec='inf', size_data=1, save=False, save_path=''):
    data = {}
    BBlist, BDlist, DDlist, Colist, Blist, Clist, Dlist, nBlist, Ilist, \
    nonIlist, BBdepthlist, BDdepthlist, DDdepthlist, Breglist, Dreglist ,\
    BBdepthreglist, BDdepthreglist, DDdepthreglist, startBlist, startClist = stat
    
    data['branch per Neural depth'] = Blist.astype(float)/size_data
    data['extension per Neural depth'] = Clist.astype(float)/size_data
    data['terminal per Neural depth'] = Dlist.astype(float)/size_data
    data['branch per branch depth'] = Breglist.astype(float)/size_data
    data['terminal per branch depth'] = Dreglist.astype(float)/size_data
    
    data['++ per Neural depth'] = BBdepthlist.astype(float)/size_data
    data['+- per Neural depth'] = BDdepthlist.astype(float)/size_data
    data['-- per Neural depth'] = DDdepthlist.astype(float)/size_data
    data['++ per branch depth'] = BBdepthreglist.astype(float)/size_data
    data['+- per branch depth'] = BDdepthreglist.astype(float)/size_data
    data['-- per branch depth'] = DDdepthreglist.astype(float)/size_data
    
    data['asymmetric ratio neural'] = asy_ratio(BBdepthlist, BDdepthlist, DDdepthlist)
    data['asymmetric ratio branch'] = asy_ratio(BBdepthreglist, BDdepthreglist, DDdepthreglist)
    data['initial outgoing segments with branch'] = float(Ilist)/size_data
    data['non-trivial initials outgoing segments'] = float(nonIlist)/size_data
    data['outgoing segments starting imediatly with branch'] = float(startBlist)/size_data
    data['outgoing segments starting imediatly with extension'] = float(startClist)/size_data
    data['++ '] = float(BBlist)/size_data
    data['+-'] = float(BDlist)/size_data
    data['--'] = float(DDlist)/size_data
    data['branches'] = float(nBlist)/size_data
    data['extensions'] = float(Colist)/size_data

    fig = plt.figure(figsize=(15, 8)) 
    gs = gridspec.GridSpec(2, 3) 
    ax0 = plt.subplot(gs[0])
    
    pa, = ax0.plot(cut_vec(data['branch per Neural depth'], length_stat_vec), label="Branches")
    pb, = ax0.plot(cut_vec(data['extension per Neural depth'], length_stat_vec), label="Extenstions")   
    pc, = ax0.plot(cut_vec(data['terminal per Neural depth'], length_stat_vec), label="Terminal")
    ax0.legend(handles=[pa, pb, pc])
    ax0.set_title("mean of different node types per Neural depth")

    ax1 = plt.subplot(gs[1])
    pa, = ax1.plot(cut_vec(data['branch per branch depth'], length_stat_vec), label="Branches")
    pb, = ax1.plot(cut_vec(data['terminal per branch depth'], length_stat_vec), label="Terminal")
    ax1.legend(handles=[pa, pb])
    ax1.set_title("mean of different node types per BRANCH depth")
    
    
    ax2 = plt.subplot(gs[2])
    pa, = ax2.plot(cut_vec(data['++ per Neural depth'], length_stat_vec), label="++: pth children branch")
    pb, = ax2.plot(cut_vec(data['+- per Neural depth'], length_stat_vec), label="+-: one child branches")
    pc, = ax2.plot(cut_vec(data['-- per Neural depth'], length_stat_vec), label="--: neither of children branch")
    ax2.legend(handles=[pa, pb, pc])
    ax2.set_title("mean of different branch types per NEURAL depth")

    ax3 = plt.subplot(gs[3])
    pa, = ax3.plot(cut_vec(data['++ per branch depth'], length_stat_vec), label="++: pth children branch")
    pb, = ax3.plot(cut_vec(data['+- per branch depth'], length_stat_vec), label="+-: one child branches")
    pc, = ax3.plot(cut_vec(data['-- per branch depth'], length_stat_vec), label="--: neither of children branch")
    ax3.legend(handles=[pa, pb, pc])
    ax3.set_title("mean of different branch types per BRANCH depth")


    ax4 = plt.subplot(gs[4])
    pa, = ax4.plot(cut_vec(data['asymmetric ratio neural'], length_stat_vec), label="Neural subsample")
    pb, = ax4.plot(cut_vec(data['asymmetric ratio branch'], length_stat_vec), label="Branch subsample")
    ax4.legend(handles=[pa, pb])
    ax4.set_title("Asymmetric ratio per depth")


    ax5 = plt.subplot(gs[5])
    width=.35
    ind = np.arange(9)
    rects1 = ax5.bar(ind,
                    np.array([float(Ilist)/size_data,
                              float(nonIlist)/size_data,
                              float(startBlist)/size_data,
                              float(startClist)/size_data,
                              float(nBlist)/size_data,
                              float(Colist)/(100*size_data),
                              float(BBlist)/size_data, 
                              float(BDlist)/size_data, 
                              float(DDlist)/size_data]), 
                    width=width, 
                    color='r')
    ax5.set_ylabel('Number')
    ax5.set_title('Scalar Statistics (averaged)')
    ax5.set_xticks(ind)
    ax5.set_xticklabels(('initial outgoing segments with branch',
                        'non-trivial initials outgoing segments',
                        'outgoing segments starting imediatly with branch', 
                        'outgoing segments starting imediatly with extension', 
                        'branches',
                        'extensions * 100',
                        '++ ', 
                        '+-',
                        '--'), rotation='vertical')

    if save:
        plt.savefig(save_path+"statistics.eps")
    plt.show()
    return data

def get_values(stat,
               indices='all',
               k_folds=10):
    ex_stat = {}
    ex_stat['subset_s'] = subset_stat(stat, indices)
    ex_stat['para'] = fit(ex_stat['subset_s'])
    if k_folds is not 0:
        ex_stat['likelihoods'] = cross_validation(stat, indices=indices, k_folds=k_folds)
    return ex_stat
    
def show_database(data,
                  ex_stat,
                  indices='all',
                  length_stat_vec='inf',
                  show_n_sample=1,
                  show_neuron=True, 
                  subsample_length='epsilon',
                  save=False,
                  save_path='',
                  part='all'):
    
    subset_s = ex_stat['subset_s']
    data_stat = show_stat(subset_s, 
                          length_stat_vec=length_stat_vec,
                          size_data=len(indices),
                          save=save,
                          save_path=save_path)
    para = ex_stat['para']
    data_fit = show_fit(para, 
                        length_stat_vec=length_stat_vec,
                        save=save,
                        save_path=save_path)

    fig = plt.figure(figsize=(12,8)) 
    gs = gridspec.GridSpec(show_n_sample,3) 
    print("Samples:")
    for i in range(show_n_sample):
  
        index = np.random.choice(indices,1)[0]
        if subsample_length is 'epsilon':
            swc = data[index]
        else:
            swc = subsample.fast_straigthen_subsample_swc(select_part_swc(swc_matrix=data[index], part=part), length=subsample_length)
        neuron = Neuron(swc)
        sub_neuron = \
            Neuron(subsample.regular_subsample(select_part_swc(swc_matrix=data[index], part=part)))
                
        if show_neuron:
            ax = plt.subplot(gs[3*i])
            visualize.plot_2D(neuron, pass_ax=True, ax=ax)
        ax = plt.subplot(gs[3*i+1])
        visualize.plot_dendrogram(neuron, pass_ax=True, ax=ax)
        ax = plt.subplot(gs[3*i+2])
        visualize.plot_dendrogram(sub_neuron, pass_ax=True, ax=ax)
    #likelihoods = cross_validation(stat, indices=indices, k_folds=k_folds)
    show_ML_models(ex_stat['likelihoods'],
                   save=save,
                   save_path=save_path)
    if save:
        plt.savefig(save_path+"neurons.eps")  
    return data_stat, data_fit, ex_stat['likelihoods']

def sum_(v1, v2):
    """
    Summation of two vectors with not equal size.
    """
    summation = np.zeros(max(len(v1), len(v2)))
    summation[0:len(v1)] = v1
    summation[0:len(v2)] = summation[0:len(v2)] + v2
    return summation

def select_part_swc(swc_matrix, part='all'):
    """
    Return 0 if the selection is not possible.
    """
    if part == 'all':
        return swc_matrix
    elif part == 'axon':
        return neuron_with_node_type(swc_matrix, index=[2])        
    elif part == 'basal':
        return neuron_with_node_type(swc_matrix, index=[3])        
    elif part == 'apical':
        return neuron_with_node_type(swc_matrix, index=[4])  
    elif part == 'dendrite':
        return neuron_with_node_type(swc_matrix, index=[3,4])  

    
def neuron_with_node_type(swc_matrix, index):
    if(swc_matrix.shape[0]==0):
        return swc_matrix
    else:
        swc_matrix[0,1] = 1
        swc_matrix[swc_matrix[:,6]==-1,6] = 1
        (soma,) = np.where(swc_matrix[:, 1] == 1)
        all_ind = [np.where(swc_matrix[:, 1] == i)[0] for i in index]
        l = sum([np.sign(len(i)) for i in all_ind])
        nodes = np.sort(np.concatenate(all_ind))
        subset = np.sort(np.append(soma, nodes))
        labels_parent = np.unique(swc_matrix.astype(int)[swc_matrix.astype(int)[subset,6],1])
        labels_parent = np.sort(np.delete(labels_parent, np.where(labels_parent==1)))
        if len(nodes) == 0 or ~np.all(np.in1d(labels_parent, index)):
            return 0 
        else:
            le = preprocessing.LabelEncoder()
            le.fit(subset)
            parent = le.transform(swc_matrix[subset[1:],6].astype(int))
            new_swc = swc_matrix[subset,:]
            new_swc[1:,6] = parent
            return new_swc

def get_stat_for_single_neuron(swc_matrix, length='epsilon', part='all'):
    """
    Get Statistics for a single tree.
    """
    swc_matrix = select_part_swc(swc_matrix, part=part)
    if swc_matrix is 0:
        straigthen_swc = np.array([[1,1,0,0,0,1,-1],[2,1,0,0,1,1,1]]) 
        reg_swc = np.array([[1,1,0,0,0,1,-1],[2,1,0,0,1,1,1]])
    else:
        if(length=='epsilon'):     
            straigthen_swc = swc_matrix
            reg_swc = subsample.regular_subsample(swc_matrix)
        else:
            straigthen_swc = subsample.fast_straigthen_subsample_swc(swc_matrix, length)
            reg_swc = subsample.regular_subsample(swc_matrix)
    

    neuron = Neuron(straigthen_swc)
    neuron.motif_features()
    neuron.basic_features()
    neuron_reg = Neuron(reg_swc)
    neuron_reg.motif_features()
    neuron_reg.basic_features()
    
    BB = neuron.features['branch branch'][0]
    BD = neuron.features['branch die'][0]
    DD = neuron.features['die die'][0]
    
    Co = neuron.features['Npassnode'][0]
    B = neuron.features['branch depth']
    Breg = neuron_reg.features['branch depth']
    C = neuron.features['continue depth']
    D = neuron.features['dead depth']
    Dreg = neuron_reg.features['dead depth']
    nB = neuron.features['Nbranch'][0] - (neuron.features['initial with branch'][0]+1)
    I = neuron.features['initial with branch'][0]+1
    nonI = neuron.features['all non trivial initials'][0]
    BBdepth = neuron.features['branch branch depth']
    BBdepthreg = neuron_reg.features['branch branch depth']
    BDdepth = neuron.features['branch die depth']
    BDdepthreg = neuron_reg.features['branch die depth']
    DDdepth = neuron.features['die die depth']
    DDdepthreg = neuron_reg.features['die die depth']
    I = float(I)
    nB = float(nB)
    BB = float(BB)
    BD = float(BD)
    DD = float(DD)
    B = B.astype(float)
    
    return BB, BD, DD, Co, B, C, D,  nB, I, nonI, BBdepth, BDdepth, DDdepth, Breg, Dreg , BBdepthreg, BDdepthreg, DDdepthreg 

def asy_ratio(BB, BD, DD):
    return (BD*BD).astype(float)/(4*BB*DD).astype(float)

# sample from models
def array_cliping(array, index, cliping, value):
    if index < cliping:
        return array[index]
    else:
        return value

# Symmetric
def sample_sym(pb, pc, cliping=100, initials=2):
    """
    In th symmetric model, at the starting time there are a few
    initial segment which are active. Then in each time, the active
    node may branch, continoue or die independently based on
    the probability which depends on the time (or steps). The active nodes
    became inactive once the time passes.
    """
    parent = np.zeros(initials+1)
    active = np.ones(initials+1)
    active[0] = 0
    depth = np.ones(initials+1, dtype=int)
    depth[0] = 0

    while active.sum():
        (ac,) = np.where(active)
        ac = ac[0]
        r = np.random.rand()
        d = depth[ac]
        b_prob = array_cliping(pb, d, cliping, 0)
        c_prob = array_cliping(pc, d, cliping, 0)
        if r < b_prob:
            parent = np.append(parent, np.array([ac, ac]))
            active = np.append(active, np.array([1, 1]))
            depth = np.append(depth, np.array([d + 1, d + 1]))
        elif r < c_prob + b_prob:
            parent = np.append(parent, np.array([ac]))
            active = np.append(active, np.array([1]))
            depth = np.append(depth, np.array([d + 1]))
        active[ac] = 0
    parent += 1
    parent[0] = 0
    return parent

# Symmetric
def sample_asym(pa, pb, pc, cliping=100, initials=2):
    parent = np.zeros(initials+1)
    active = np.ones(initials+1)
    active[0] = 0
    depth = np.ones(initials+1, dtype=int)
    depth[0] = 0
    node_type = np.zeros(initials+1)
    while active.sum():
        (ac,) = np.where(active)
        ac = ac[0]
        r = np.random.rand()
        d = depth[ac]
        c_prob = array_cliping(pc, d, cliping, 0)
        n_type = node_type[ac]
        if n_type == 0:
            b_prob = array_cliping(pa, d, cliping, 0)
        else:
            b_prob = array_cliping(pb, d, cliping, 0)
        if r < b_prob:
            parent = np.append(parent, np.array([ac, ac]))
            active = np.append(active, np.array([1, 1]))
            depth = np.append(depth, np.array([d + 1, d + 1]))
            node_type = np.append(node_type, np.array([0, 1]))
        elif r < c_prob + b_prob:
            parent = np.append(parent, np.array([ac]))
            active = np.append(active, np.array([1]))
            depth = np.append(depth, np.array([d + 1]))
            node_type = np.append(node_type, n_type)
        active[ac] = 0
    parent += 1
    parent[0] = 0
    return parent

def sample_complete_asym(pa_aa, pa_ab, pa_bb, pb_aa, pb_ab, pb_bb, pc,
                         cliping=100,
                         value_cliping=0,
                         initials=2):
    parent = np.zeros(initials+1)
    active = np.ones(initials+1)
    active[0] = 0
    depth = np.ones(initials+1, dtype=int)
    depth[0] = 0
    node_type = (initials+1)*['a']
    while active.sum():
        (ac,) = np.where(active)
        ac = ac[0]
        r = np.random.rand()
        d = depth[ac]
        c_prob = array_cliping(pc, d, cliping, value_cliping)
        n_type = node_type[ac]
        if n_type == 'a':
            #print('node is a')
            aa_prob = array_cliping(array=pa_aa, index=d, cliping=cliping, value=value_cliping)
            ab_prob = array_cliping(pa_ab, d, cliping, value_cliping)
            bb_prob = array_cliping(pa_bb, d, cliping, value_cliping)
        else:
            #print('node is b')
            aa_prob = array_cliping(pb_aa, d, cliping, value_cliping)
            ab_prob = array_cliping(pb_ab, d, cliping, value_cliping)
            bb_prob = array_cliping(pb_bb, d, cliping, value_cliping)
        if r < aa_prob:
            #print('aa')
            parent = np.append(parent, np.array([ac, ac]))
            active = np.append(active, np.array([1, 1]))
            depth = np.append(depth, np.array([d + 1, d + 1]))
            node_type.append('a')
            node_type.append('a')
        elif r < aa_prob + ab_prob:
            #print('ab')
            parent = np.append(parent, np.array([ac, ac]))
            active = np.append(active, np.array([1, 1]))
            depth = np.append(depth, np.array([d + 1, d + 1]))
            node_type.append('a')
            node_type.append('b')
        elif r < aa_prob + ab_prob + bb_prob:
            #print('bb')
            parent = np.append(parent, np.array([ac, ac]))
            active = np.append(active, np.array([1, 1]))
            depth = np.append(depth, np.array([d + 1, d + 1]))
            node_type.append('b')
            node_type.append('b')
        elif r < aa_prob + ab_prob + bb_prob + c_prob:
            #print('c')
            parent = np.append(parent, np.array([ac]))
            active = np.append(active, np.array([1]))
            depth = np.append(depth, np.array([d + 1]))
            node_type.append(n_type)
        active[ac] = 0

    parent += 1
    parent[0] = 0
    return parent

def generate_neurons(model, para, size, initials=2, cliping=100):
    data = []
    pb = para['pb']
    pc = para['pc']
    while len(data) < size:
        if model == 'sym':
            parent = sample_sym(pb, pc, initials=initials, cliping=cliping)
        elif model == 'asym':
            pa = para['pa']
            parent = sample_asym(pa, pb, pc, initials=initials, cliping=cliping)
        if(parent.shape[0]>1):
            A = np.random.rand(parent.shape[0],7)
            A[0,1] = 1
            A[0,0] = 1
            A[:,6] = parent

            # n = Neuron(input_file=A)
            data.append(A)
    return data

def generate_trees(model, para, size, initials=2, cliping=100):
    data = []
    while len(data) < size:
        if model == 'sym':
            parent = sample_sym(pb=para['pb'],
                                pc=para['pc'],
                                initials=initials,
                                cliping=cliping)
        elif model == 'asym':
            parent = sample_asym(pa = para['pa'],
                                 pb = para['pb'],
                                 pc = para['pc'],
                                 initials=initials,
                                 cliping=cliping)
        elif model == 'general':
            parent = sample_complete_asym(pa_aa=para['paaa'],
                                          pa_ab=para['paab'],
                                          pa_bb=para['pabb'],
                                          pb_aa=para['pbaa'],
                                          pb_ab=para['pbab'],
                                          pb_bb=para['pbbb'],
                                          pc=para['pc'],
                                          cliping=cliping,
                                          value_cliping=0,
                                          initials=initials)
        if(parent.shape[0]>1):
            p = parent.astype(int) -1
            p[0] = 0
            data.append(p)
    return data

# All the likelihood are calculated by the assumption that the root
# has one non-trivial extension (branching or extension)

def log_(a):
    """avoid zeros"""
    if a <=0:
        return 0
    else:
        return np.log(a)

def ML_sym(nbranch, ntree):
    """
    Parameters
    ----------
    nbranch: int
        The number of branches
    ntree: int
        the number of trees (non-trivials)
    Return
    ------
    p: float (probability value)
        the best fitting value for the simple symmetric model
    """
    p = float(nbranch)/(2*float(nbranch)+2*float(ntree))
    return p

def likelihood_sym(p, nB, I):
    return -(nB*log_(p) + (nB+2*I)*log_(1-p))

def ML_sym_with_C(nB, Co, I, nonI, startB, startC):
    branches = nB + I
    terminals = nB + I + nonI
    extensions = Co
    dominator = \
        terminals*((branches+extensions)/(branches+extensions - startB - startC)) + branches + extensions
    pb = branches/dominator
    pc = extensions/dominator
    return pb, pc

def likelihood_sym_with_C(x, nB, Co, I, nonI, startB, startC):
    pb, pc = x
    return -((nB+startB)*log_(pb) + Co*log_(pc) + (nB+I+nonI)*log_(1-pb-pc) - (startB+startC)*log_(pb+pc))

def ML_sym_depth(B, D):
    I = B + D
    I = I.astype(float)
    pb = B.astype(float)/I
    pd = D.astype(float)/I
    return pb, pd

def likelihood_sym_depth(x, B, D):
    pb, pd = x
    IB = pb!=0
    ID = pd!=0
    l = sum(B[IB]*np.log(pb[IB]))  + sum(D[ID]*np.log(pd[ID]))
    return -l

def ML_sym_depth_with_C(B, C, D):
    I = B + C + D
    I = I.astype(float)
    p_b = B.astype(float)/I
    p_c = C.astype(float)/I
    p_d = D.astype(float)/I
    p_b[0] = B[0]/(B[0]+C[0])
    p_c[0] = C[0]/(B[0]+C[0])
    p_d[0] = 0
    return p_b[:-1], p_c[:-1], p_d[:-1]


def likelihood_sym_depth_with_C(x, B, C, D):
    p_bC = x[0]
    p_cC = x[1]
    p_dC = x[2]
    IB = p_bC!=0
    IC = p_cC!=0
    ID = p_dC!=0
    l = sum(B[IB]*np.log(p_bC[IB])) + \
        sum(C[IC]*np.log(p_cC[IC])) + \
        sum(D[ID]*np.log(p_dC[ID]))
    return -l

def ML_asym(BB, BD, DD):
    """
    Parameters
    ----------
    BB: int
        The number of branchings that both of its children are branching.
    BD: int
        The number of branchings that one of its child is branching
        and another is leaf.
    DD: int
        The number of branchings that both of its children are leaf.
    Return
    ------
    pa, pb: float (probability value)
        the best fitting values for the simple asymmetric model
    l: float
        likelihood of the model
    """
    pa, pb = grid_search_asym(BB,BD,DD)
    return pa, pb

def likelihood_asym_simple(x, BB, BD, DD):
    l = BD*np.log(x[0]*(1-x[1])+x[1]*(1-x[0])) + \
        BB*(np.log(x[0]) + np.log(x[1])) + \
        DD*(np.log(1-x[0]) + np.log(1-x[1])) \
        - BD*np.log(2)
    return -l

def ML_asym_with_C(BB, BD, DD, Co, startB, startC):
    pa, pb, pc = grid_search_asym_with_C(BB, BD, DD, Co, startB, startC)
    return pa, pb, pc

def likelihood_asym_with_C(x, BB, BD, DD, Co, startB, startC):
    # l = + BD * np.log(x[0]*(1-x[1]-x[2])+x[1]*(1-x[0]-x[2])) \
    #     + BB * (np.log(x[0]) + np.log(x[1])) \
    #     + DD * (np.log(1-x[0]-x[2]) + np.log(1-x[1]-x[2])) \
    #     + Co * np.log(x[2]) \
    #     - nonI * np.log(x[0]+x[2]) \
    #     + (nonI - I) * np.log(1-x[0]-x[2]) \
    #     + I * np.log(x[0]) \
    #     - BD * np.log(2)
    l = + BD * np.log(x[0]*(1-x[1]-x[2])+x[1]*(1-x[0]-x[2])) \
        + BB * (np.log(x[0]) + np.log(x[1])) \
        + DD * (np.log(1-x[0]-x[2]) + np.log(1-x[1]-x[2])) \
        + (Co-startC) * np.log(x[2]) \
        - BD * np.log(2)\
        + startB * log_(x[0])\
        + startC * log_(x[2])\
        - (startC+startB) * log_(x[2]+x[0])
    return -l

def grid_search_asym_with_C(BB, BD, DD, Co, startB, startC):
    a = np.arange(0.01, 1, .01)
    b = np.arange(0.01, 1, .01)
    c = np.arange(0.01, 1, .01)

    m1, m2, m3= np.meshgrid(a,b,c) 
    length_all = len(a)*len(b)*len(c)
    m1 = np.reshape(m1,newshape=(length_all,))
    m2 = np.reshape(m2,newshape=(length_all,))
    m3 = np.reshape(m3,newshape=(length_all,))
    cond1 = np.where(1 - m1 - m3 >0)
    cond2 = np.where(1 - m2 - m3 >0)
    cond = np.intersect1d(cond1, cond2)
    m1 = m1[cond]
    m2 = m2[cond]
    m3 = m3[cond]
    l = BD* np.log(m1*(1-m2-m3)+m2*(1-m1-m3)) + \
        BB* (np.log(m1) + np.log(m2)) + \
        DD* (np.log(1-m1-m3) + np.log(1-m2-m3)) + \
        (Co-startC)* np.log(m3) - \
        + startB * np.log(m1)\
        + startC * np.log(m3)\
        - (startC+startB) * np.log(m1+m3)
    
    arg = np.argmax(l)
    a, b, pc = fmin(likelihood_asym_with_C,
                        x0 = np.array([m1[arg], m2[arg], m3[arg]]),
                        args=(BB, BD, DD, Co, startB, startC),
                        disp=False) 
    pa, pb = max(a, b), min(a, b)
    return pa, pb, pc

def ML_asym_depth(BBdepth, BDdepth, DDdepth):
    n = BBdepth.shape[0]
    pa = np.zeros(n)
    pb = np.zeros(n)
    for i in range(n):
        a, b = grid_search_asym(BBdepth[i], BDdepth[i], DDdepth[i])
        a, b = max(a, b), min(a, b)
        pa[i] = a
        pb[i] = b
    return pa, pb

def grid_search_asym(BB,BD,DD):
    a = np.arange(0.01, 1, .01)
    b = np.arange(0.01, 1, .01)

    m1, m2 = np.meshgrid(a,b) 
    length_all = len(a)*len(b)
    m1 = np.reshape(m1,newshape=(length_all,))
    m2 = np.reshape(m2,newshape=(length_all,))

    l = BD* np.log(m1*(1-m2)+m2*(1-m1)) + \
        BB* (np.log(m1) + np.log(m2)) + \
        DD* (np.log(1-m1) + np.log(1-m2))
    arg = np.argmax(l)

    a, b = fmin(likelihood_asym_simple,
                      x0 = np.array([m1[arg], m2[arg]]),
                      args=(BB, BD, DD),
                      disp=False) 
    pa, pb = max(a, b), min(a, b)
    return pa, pb

def likelihood_asym_depth(x, BBdepthreg, BDdepthreg, DDdepthreg):
    l = 0
    for i in range(len(BBdepthreg)):
        l += BDdepthreg[i]* log_(x[0][i]*(1-x[1][i])+x[1][i]*(1-x[0][i])) + \
             BBdepthreg[i]* (log_(x[0][i]) + log_(x[1][i])) + \
             DDdepthreg[i]* (log_(1-x[0][i]) + log_(1-x[1][i])) - \
            BDdepthreg[i]* np.log(2)
    return -l

def ML_asym_depth_with_C(BBdepth, BDdepth, DDdepth, C, startB, startC):
    n = len(BBdepth)
    pa = np.zeros(n)
    pb = np.zeros(n)
    pc = np.zeros(n)
    pa[0] = startB/(startB + startC)
    pc[0] = startC/(startB + startC)
    for i in range(1,n):
        a, b, c = grid_search_asym_depth_with_C(BBdepth[i-1], BDdepth[i-1], DDdepth[i-1], C[i])
        pa[i] = a
        pb[i] = b
        pc[i] = c

    return pa, pb, pc

def grid_search_asym_depth_with_C(BB,BD,DD,C):
    a = np.arange(0.01, 1, .01)
    b = np.arange(0.01, 1, .01)
    c = np.arange(0.01, 1, .01)

    m1, m2, m3= np.meshgrid(a,b,c) 
    length_all = len(a)*len(b)*len(c)
    m1 = np.reshape(m1,newshape=(length_all,))
    m2 = np.reshape(m2,newshape=(length_all,))
    m3 = np.reshape(m3,newshape=(length_all,))
    cond1 = np.where(1 - m1 - m3 >0)
    cond2 = np.where(1 - m2 - m3 >0)
    cond = np.intersect1d(cond1, cond2)
    m1 = m1[cond]
    m2 = m2[cond]
    m3 = m3[cond]
    l = BD* np.log(m1*(1-m2-m3)+m2*(1-m1-m3)) + \
        BB* (np.log(m1) + np.log(m2)) + \
        DD* (np.log(1-m1-m3) + np.log(1-m2-m3)) + \
        C* np.log(m3)
    arg = np.argmax(l)

    a, b, pc = fmin(_likelihood_asym_depth_with_C,
                        x0 = np.array([m1[arg], m2[arg], m3[arg]]),
                        args=(BB, BD, DD, C),
                        disp=False) 
    pa, pb = max(a, b), min(a, b)
    return pa, pb, pc

def _likelihood_asym_depth_with_C(x, BB, BD, DD, C):
    l = BD* np.log(x[0]*(1-x[1]-x[2])+x[1]*(1-x[0]-x[2])) + \
        BB* (np.log(x[0]) + np.log(x[1])) + \
        DD* (np.log(1-x[0]-x[2]) + np.log(1-x[1]-x[2])) + \
        C* np.log(x[2]) - \
        BD*np.log(2)
    return -l
        
def likelihood_asym_depth_with_C(x, BBdepth, BDdepth, DDdepth, C, startB, startC):
    n = len(BBdepth)
    l = -startB*log_(x[0][0]) - startC*log_(x[2][0])
    for i in range(1,n):
        l += _likelihood_asym_depth_with_C([x[0][i], x[1][i], x[2][i]],
                                          BBdepth[i-1],BDdepth[i-1], DDdepth[i-1], C[i])
    return l

def extinction(pa_aa, pa_ab, pa_bb, pb_aa, pb_ab, pb_bb, pc):
    eigvals, _ = np.linalg.eig(np.array([[2*pa_aa+pa_ab+pc, 2*pb_aa+pb_ab],
                        [pa_ab+2*pa_bb,2*pb_bb+pb_ab+pc]]))
    #print eigvals
    if(all(eigvals <1)):
        return 'extinction'
    else:
        return 'no extinction'


def cross_validation(stat,
                     indices='all',
                     k_folds=10):
    if indices is 'all':
        indices = np.arange(len(stat[0]))
    likelihood = np.array([])
    ind = deepcopy(indices)
    np.random.shuffle(ind)
    len_fold = int(float(len(ind))/float(k_folds))
    for i in range(k_folds):
        training_indices = \
            np.append(ind[:i*len_fold], ind[(i+1)*len_fold:]).astype(int)
        test_indices = ind[i*len_fold:(i+1)*len_fold] 
        para = fit(subset_stat(stat, training_indices))  
        likel = score(para, subset_stat(stat, test_indices))
        if len(likelihood)==0:
            likelihood = np.expand_dims(np.array(likel),axis=1)
        else:
            likelihood = np.append(likelihood, np.expand_dims(np.array(likel),axis=1), axis=1)
    return likelihood

def show_ML_models(likelihood,
                   save=False,
                   save_path=''):
    width=.35
    ind = np.arange(4)
    fig = plt.figure(figsize=(12, 24)) 
    gs = gridspec.GridSpec(4, 2) 
    ax = plt.subplot(gs[0,0])  
    l = likelihood - np.dot( np.ones([8,1]),np.expand_dims(likelihood[0,:],axis=0))
    rects1 = ax.bar(ind,
                    l.mean(axis=1)[:4], 
                    width=width, 
                    color='r',
                    yerr=l.std(axis=1)[:4])
    ax.set_ylabel('-Log likelihood')
    ax.set_title('compare to symetric without ext without depth')
    ax.set_xticks(ind)
    # ax.set_xticklabels(('sym no-ext',
    #                     'asym no-ext',
    #                     'sym no-ext with depth', 
    #                     'asym no-ext with depth'), 
    #                      rotation='vertical')
    
    ax = plt.subplot(gs[0,1])
    l = likelihood - np.dot( np.ones([8,1]),np.expand_dims(likelihood[1,:],axis=0))
    rects1 = ax.bar(ind,
                    l.mean(axis=1)[:4], 
                    width=width, 
                    color='r',
                    yerr=l.std(axis=1)[:4])
    ax.set_ylabel('-Log likelihood')
    ax.set_title(' compare to asymetric without ext without depth')
    ax.set_xticks(ind)
    # ax.set_xticklabels(('sym no-ext',
    #                     'asym no-ext',
    #                     'sym no-ext with depth', 
    #                     'asym no-ext with depth'), 
    #                      rotation='vertical')
    
    ax = plt.subplot(gs[1,0])
    l = likelihood - np.dot( np.ones([8,1]),np.expand_dims(likelihood[2,:],axis=0))
    rects1 = ax.bar(ind,
                    l.mean(axis=1)[:4], 
                    width=width, 
                    color='r',
                    yerr=l.std(axis=1)[:4])
    ax.set_ylabel('-Log likelihood')
    ax.set_title('compare to symetric without ext wit depth')
    ax.set_xticks(ind)
    ax.set_xticklabels(('sym no-ext',
                        'asym no-ext',
                        'sym no-ext with depth', 
                        'asym no-ext with depth'), 
                         rotation='vertical')
    
    ax = plt.subplot(gs[1,1])
    l = likelihood - np.dot( np.ones([8,1]),np.expand_dims(likelihood[3,:],axis=0))
    rects1 = ax.bar(ind,
                    l.mean(axis=1)[:4], 
                    width=width, 
                    color='r',
                    yerr=l.std(axis=1)[:4])
    ax.set_ylabel('-Log likelihood')
    ax.set_title('compare to asymetric without ext wit depth')
    ax.set_xticks(ind)
    ax.set_xticklabels(('sym no-ext',
                        'asym no-ext',
                        'sym no-ext with depth', 
                        'asym no-ext with depth'), 
                         rotation='vertical')

    
    ax = plt.subplot(gs[2,0])  
    l = likelihood - np.dot( np.ones([8,1]),np.expand_dims(likelihood[4,:],axis=0))
    rects1 = ax.bar(ind,
                    l.mean(axis=1)[4:], 
                    width=width, 
                    color='r',
                    yerr=l.std(axis=1)[4:])
    ax.set_ylabel('-Log likelihood')
    ax.set_title('compare to symetric with ext without depth')
    ax.set_xticks(ind)
    # ax.set_xticklabels(('sym with ext', 
    #                     'asym with ext', 
    #                     'sym ext with depth',
    #                     'asym ext with depth'), 
    #                      rotation='vertical')
    
    ax = plt.subplot(gs[2,1])
    l = likelihood - np.dot( np.ones([8,1]),np.expand_dims(likelihood[5,:],axis=0))
    rects1 = ax.bar(ind,
                    l.mean(axis=1)[4:], 
                    width=width, 
                    color='r',
                    yerr=l.std(axis=1)[4:])
    ax.set_ylabel('-Log likelihood')
    ax.set_title('compare to asymetric with ext without depth')
    ax.set_xticks(ind)
    # ax.set_xticklabels(('sym with ext', 
    #                     'asym with ext', 
    #                     'sym ext with depth',
    #                     'asym ext with depth'), 
    #                      rotation='vertical')
    
    ax = plt.subplot(gs[3,0])
    l = likelihood - np.dot( np.ones([8,1]),np.expand_dims(likelihood[6,:],axis=0))
    rects1 = ax.bar(ind,
                    l.mean(axis=1)[4:], 
                    width=width, 
                    color='r',
                    yerr=l.std(axis=1)[4:])
    ax.set_ylabel('-Log likelihood')
    ax.set_title('compare to symetric with ext wit depth')
    ax.set_xticks(ind)
    ax.set_xticklabels(('sym with ext', 
                        'asym with ext', 
                        'sym ext with depth',
                        'asym ext with depth'), 
                         rotation='vertical')
    
    ax = plt.subplot(gs[3,1])
    l = likelihood - np.dot( np.ones([8,1]),np.expand_dims(likelihood[7,:],axis=0))
    rects1 = ax.bar(ind,
                    l.mean(axis=1)[4:], 
                    width=width, 
                    color='r',
                    yerr=l.std(axis=1)[4:])
    ax.set_ylabel('-Log likelihood')
    ax.set_title('compare to asymetric with ext wit depth')
    ax.set_xticks(ind)
    ax.set_xticklabels(('sym with ext', 
                        'asym with ext', 
                        'sym ext with depth',
                        'asym ext with depth'), 
                         rotation='vertical')
    if save:
        plt.savefig(save_path+"cross-validations.eps")  
    
    plt.show()  

def decompose_to_outgoing_trees(parent_index):
    """
    Return
    ------
    list_trees: the list of all outgoing trees, dismissed the size ones
    """
    
    (initials,) = np.where(parent_index[1:]==0)
    initials += 1
    n_initials = len(initials)
    all_outgoing_trees = []
    for i in range(n_initials):
        all_outgoing_trees.append([0])
    which_outgoing_tree = np.zeros(len(parent_index))
    which_outgoing_tree[initials] = np.arange(n_initials)
    index_in_outgoing_tree = np.zeros(len(parent_index))
    index_node = 0
    for index_parent in parent_index:
        if(index_parent!=0):
            which_tree = int(which_outgoing_tree[index_parent])            
            index_in_outgoing_tree[index_node] = len(all_outgoing_trees[which_tree])
            which_outgoing_tree[index_node] = which_tree
            all_outgoing_trees[which_tree].append(int(index_in_outgoing_tree[index_parent])) 
        index_node += 1
    for i in range(n_initials):
        all_outgoing_trees[i] = np.array(all_outgoing_trees[i])        
    return all_outgoing_trees

def decomposed_tree(tree_data):
    trees = []
    for parent_index in tree_data:
        trees = trees + decompose_to_outgoing_trees(parent_index)
    return trees

def update_dic_vector(params, dic, depth, tree, depths, same_matrix):
    (nodes,) = np.where(depths == depth)
    for node in nodes:
        (child,) = np.where(tree[1:]==node)
        if len(child) == 0:
            dic[(node, 'a')] = params['pa']
            dic[(node, 'b')] = params['pb']
        
        else:
            child += 1
            same = same_matrix[child[0], child[1]]
            dic[(node, 'a')] = experssion_decimal(pxaa=params['paaa'],
                                                  pxab=params['paab'],
                                                  pxbb=params['pabb'], 
                                                  zero_a=dic[(child[0], 'a')],
                                                  one_a=dic[(child[1], 'a')],
                                                  zero_b=dic[(child[0], 'b')],
                                                  one_b=dic[(child[1], 'b')],
                                                  same=same)
            
            dic[(node, 'b')] = experssion_decimal(pxaa=params['pbaa'],
                                                  pxab=params['pbab'],
                                                  pxbb=params['pbbb'], 
                                                  zero_a=dic[(child[0], 'a')],
                                                  one_a=dic[(child[1], 'a')],
                                                  zero_b=dic[(child[0], 'b')],
                                                  one_b=dic[(child[1], 'b')],
                                                  same=same)
    return dic

def experssion_decimal(pxaa, pxab, pxbb, zero_a, one_a, zero_b, one_b, same):
    """simplified the expression that should be used recursively for nodes
    """
    if same==1:
        return [pxaa_lit*zero_a_lit*one_a_lit+ \
                Decimal(.5)*pxab_lit*(zero_a_lit*one_b_lit+zero_b_lit*one_a_lit)+\
                pxbb_lit*zero_b_lit*one_b_lit \
                for pxaa_lit, pxab_lit, pxbb_lit, zero_a_lit, one_a_lit, zero_b_lit, one_b_lit \
                in zip(pxaa, pxab, pxbb, zero_a, one_a, zero_b, one_b)]
    else:
        return [Decimal(2)*pxaa_lit*zero_a_lit*one_a_lit+ \
                pxab_lit*(zero_a_lit*one_b_lit+zero_b_lit*one_a_lit)+\
                Decimal(2)*pxbb_lit*zero_b_lit*one_b_lit \
                for pxaa_lit, pxab_lit, pxbb_lit, zero_a_lit, one_a_lit, zero_b_lit, one_b_lit \
                in zip(pxaa, pxab, pxbb, zero_a, one_a, zero_b, one_b)]

def mesh_decimal(paaa_min, paaa_max, paaa_mesh,
                 paab_min, paab_max, paab_mesh,
                 pabb_min, pabb_max, pabb_mesh,
                 pbaa_min, pbaa_max, pbaa_mesh,
                 pbab_min, pbab_max, pbab_mesh,
                 pbbb_min, pbbb_max, pbbb_mesh):
    """
    making the big list of decimal for all the possible probabilities
    """
    ipaaa = np.arange(paaa_min, paaa_max, paaa_mesh)
    ipaab = np.arange(paab_min, paab_max, paab_mesh)
    ipabb = np.arange(pabb_min, pabb_max, pabb_mesh)
    ipbaa = np.arange(pbaa_min, pbaa_max, pbaa_mesh)
    ipbab = np.arange(pbab_min, pbab_max, pbab_mesh)
    ipbbb = np.arange(pbbb_min, pbbb_max, pbbb_mesh)
    m1, m2, m3, m4, m5, m6 = np.meshgrid(ipaaa, ipaab, ipabb, ipbaa, ipbab, ipbbb) 
    length_all = len(ipaaa)*len(ipaab)*len(ipabb)*len(ipbaa)*len(ipbab)*len(ipbbb)
    m1 = np.reshape(m1,newshape=(length_all,))
    m2 = np.reshape(m2,newshape=(length_all,))
    m3 = np.reshape(m3,newshape=(length_all,))
    m4 = np.reshape(m4,newshape=(length_all,))
    m5 = np.reshape(m5,newshape=(length_all,))
    m6 = np.reshape(m6,newshape=(length_all,))
    cond1 = np.where(1 - m1 - m2 - m3 >0)
    cond2 = np.where(1 - m4 - m5 - m6>0)
    cond = np.intersect1d(cond1, cond2)
    m1 = m1[cond]
    m2 = m2[cond]
    m3 = m3[cond]
    m4 = m4[cond]
    m5 = m5[cond]
    m6 = m6[cond]
    
    params = {}
    params['paaa'] = [Decimal(x) for x in m1]
    params['paab'] = [Decimal(x) for x in m2]
    params['pabb'] = [Decimal(x) for x in m3]
    params['pbaa'] = [Decimal(x) for x in m4]
    params['pbab'] = [Decimal(x) for x in m5]
    params['pbbb'] = [Decimal(x) for x in m6]
    params['pa'] = [Decimal(1 - x - y - z) for x, y, z in zip(m1, m2, m3)]
    params['pb'] = [Decimal(1 - x - y - z) for x, y, z in zip(m4, m5, m6)] 
    return params
    
def logp_vector(trees, all_same_mat,
                paaa_min, paaa_max, paaa_mesh,
                paab_min, paab_max, paab_mesh,
                pabb_min, pabb_max, pabb_mesh,
                pbaa_min, pbaa_max, pbaa_mesh,
                pbab_min, pbab_max, pbab_mesh,
                pbbb_min, pbbb_max, pbbb_mesh):
    """
    trees: list
    the list of all parent_index of the trees
    """

    params = mesh_decimal(paaa_min, paaa_max, paaa_mesh,
                          paab_min, paab_max, paab_mesh,
                          pabb_min, pabb_max, pabb_mesh,
                          pbaa_min, pbaa_max, pbaa_mesh,
                          pbab_min, pbab_max, pbab_mesh,
                          pbbb_min, pbbb_max, pbbb_mesh)
    params = select_admissibles(params)
    prob_value = np.zeros(len(params['pa']))
    itr = 0
    for tree in trees:
        dic = {}
        depths = McNeuron.neuron_util.dendogram_depth(tree)
        same_matrix = all_same_mat[itr]
        for depth in range(max(depths),0,-1):
            update_dic_vector(params, dic, depth, tree, depths, same_matrix)
        (outgoings,) = np.where(depths==1)
        for out in outgoings:
            prob_value += np.array([-float(i.ln()) for i in dic[(out, 'a')]])
        itr += 1
        
    min_index = np.argmin(prob_value)
    print prob_value[min_index]
    paaa_ml = float(params['paaa'][min_index])
    paab_ml = float(params['paab'][min_index])
    pabb_ml = float(params['pabb'][min_index])
    pbaa_ml = float(params['pbaa'][min_index])
    pbab_ml = float(params['pbab'][min_index])
    pbbb_ml = float(params['pbbb'][min_index])
    return paaa_ml, paab_ml, pabb_ml, pbaa_ml, pbab_ml, pbbb_ml

def log_vector_all(tree, same_mat, params):
    prob_value = np.zeros(len(params['pa']))
    dic = {}
    depths = McNeuron.neuron_util.dendogram_depth(tree)
    for depth in range(max(depths),0,-1):
        update_dic_vector(params, dic, depth, tree, depths, same_mat)
    (outgoings,) = np.where(depths==1)
    for out in outgoings:
        prob_value += np.array([-float(i.ln()) for i in dic[(out, 'a')]])
    return prob_value
        
def set_matrix_same(trees):
    all_same_matrix = []
    for t in trees:
        all_same_matrix.append(matrix_same(tree))
    return all_same_matrix

def select_admissibles(grid_list):
    ext_list = np.array([])
    for i in range(len(grid_list['pa'])):    
        extnot = asy.extinction(float(grid_list['paaa'][i]),
                                float(grid_list['paab'][i]),
                                float(grid_list['pabb'][i]),
                                float(grid_list['pbaa'][i]),
                                float(grid_list['pbab'][i]),
                                float(grid_list['pbbb'][i]),
                               0)
        if extnot is 'extinction':
            ext_list = np.append(ext_list,i)
    admissible_list = {} 
    ext_list = ext_list.astype(int)
    admissible_list['pa'] = list(grid_list['pa'][i] for i in ext_list)
    admissible_list['pb'] = list(grid_list['pb'][i] for i in ext_list)
    admissible_list['paaa'] = list(grid_list['paaa'][i] for i in ext_list)
    admissible_list['paab'] = list(grid_list['paab'][i] for i in ext_list)
    admissible_list['pabb'] = list(grid_list['pabb'][i] for i in ext_list)
    admissible_list['pbaa'] = list(grid_list['pbaa'][i] for i in ext_list)
    admissible_list['pbab'] = list(grid_list['pbab'][i] for i in ext_list)
    admissible_list['pbbb'] = list(grid_list['pbbb'][i] for i in ext_list)
    return admissible_list  

def recursive_logp(trees, iterations, all_same_matrix):        
    aaa = (0., 1.)
    aab = (0., 1.)
    abb = (0., 1.)
    baa = (0., 1.)
    bab = (0., 1.)
    bbb = (0., 1.)
    mesh = 0.25
    for i in range(iterations):
        opt = logp_vector(trees,all_same_matrix,
                          paaa_min=aaa[0] + (mesh / 2.), paaa_max=aaa[1], paaa_mesh=mesh,
                          paab_min=aab[0] + (mesh / 2.), paab_max=aab[1], paab_mesh=mesh,
                          pabb_min=abb[0] + (mesh / 2.), pabb_max=abb[1], pabb_mesh=mesh,
                          pbaa_min=baa[0] + (mesh / 2.), pbaa_max=baa[1], pbaa_mesh=mesh,
                          pbab_min=bab[0] + (mesh / 2.), pbab_max=bab[1], pbab_mesh=mesh,
                          pbbb_min=bbb[0] + (mesh / 2.), pbbb_max=bbb[1], pbbb_mesh=mesh)
        aaa = (opt[0] - (mesh / 2.), opt[0] + (mesh / 2.))
        aab = (opt[1] - (mesh / 2.), opt[1] + (mesh / 2.))
        abb = (opt[2] - (mesh / 2.), opt[2] + (mesh / 2.))
        baa = (opt[3] - (mesh / 2.), opt[3] + (mesh / 2.))
        bab = (opt[4] - (mesh / 2.), opt[4] + (mesh / 2.))
        bbb = (opt[5] - (mesh / 2.), opt[5] + (mesh / 2.))
        mesh /= 2
        print((aaa[0] + (mesh / 2.), aab[0] + (mesh / 2.), abb[0] + (mesh / 2.), baa[0] + (mesh / 2.), bab[0] + (mesh / 2.), bbb[0] + (mesh / 2.)))
    return (aaa[0] + (mesh / 2.), aab[0] + (mesh / 2.), abb[0] + (mesh / 2.), baa[0] + (mesh / 2.), bab[0] + (mesh / 2.), bbb[0] + (mesh / 2.))

def all_same_matrix(trees):
    output = []
    for tree in trees:
        output.append(matrix_same_depth(tree))
    return output

def matrix_same(tree):
    n = len(tree)
    A = np.eye(n)
    for i in range(n):
        for j in range(i+1,n):
            same = are_same(i, j, tree) 
            A[i, j] = same
            A[j, i] = same
    return A

def matrix_same_depth(tree):
    n = len(tree)
    A = csr_matrix((n, n), dtype=np.int8)
    depths = McNeuron.neuron_util.dendogram_depth(tree)
    for depth in range(max(depths),-1,-1):     
        (I,) = np.where(depths==depth)
        for i in I:
            for j in I:
                (child1,) = np.where(tree==i)
                (child2,) = np.where(tree==j)
                if len(child1) == 0 and len(child2) == 0:
                    A[i, j] = 1
                    A[j, i] = 1
                elif len(child1) != len(child2):
                    A[i, j] = 0
                    A[j, i] = 0
                else:   
                    if (A[child1[0], child2[0]] == 1 and A[child1[1], child2[1]] == 1)\
                    or\
                    (A[child1[0], child2[1]] == 1 and A[child1[1], child2[0]] == 1):                       
                        A[i, j] = 1
                        A[j, i] = 1  
    A[0,0] = 1
    return A

def are_same(node1, node2, tree):
    (child1,) = np.where(tree==node1)
    (child2,) = np.where(tree==node2)
    if len(child1) == 0 and len(child2) == 0:
        return 1
    elif len(child1) != len(child2):
        return 0
    else:
        return (are_same(child1[0], child2[0], tree) and are_same(child1[1], child2[1], tree))\
                or \
               (are_same(child1[0], child2[1], tree) and are_same(child1[1], child2[0], tree))
            
def are_two_trees_same(tree1, tree2):
    big_tree = np.append(0,np.append(np.append(0, tree1[1:]+1),np.append(0, tree2[1:]+len(tree1)+1)))
    return are_same(1, len(tree1)+1, big_tree)

def count_motifs(dataY, neuron_trees, label, classes, trees):
    n_trees = len(trees)
    n_class = len(classes)
    count_motifs = np.zeros([n_class, n_trees])
    for ind_name in range(n_class):
        class_trees = []
        for i in range(len(dataY[label])):
            if dataY[label][i] == classes[ind_name]:
                class_trees = class_trees + decompose_to_outgoing_trees(neuron_trees[i])
        for j in range(len(trees)):
            for k in range(len(class_trees)):
                if are_two_trees_same(tree1=class_trees[k], tree2=trees[j]):
                    count_motifs[ind_name, j] += 1
        count_motifs[ind_name,:] = count_motifs[ind_name,:]/float(len(class_trees))
    return count_motifs

def show_diversity(count_motifs, trees, classes, colors=''):
    if len(colors)==0:
        colors=[]
        for i in range(len(bio_classes)):
            a = (np.mod(float(i)/5.,1),
                           np.mod(float(i)/7.,1),
                           np.mod(float(i)/11.,1))
            colors.append(a)        
    n_class = count_motifs.shape[0]
    n_trees = count_motifs.shape[1]
    fig = plt.figure(figsize=(10,8)) 
    gs = gridspec.GridSpec(3,n_trees)
    ax = plt.subplot(gs[:2,:])
    for t in range(n_trees):
        for i in range(n_class):
            ax.bar(left=t+(i/(float(1.5*n_class))),
                    height=count_motifs[i,t],
                    width=1/(float(2*n_class))
                    ,color=colors[i]
                  #,hatch=hatch[i]
                  )
            ax.set_xticks([])
    handles = []
    for i in range(len(bio_classes)):
        handles.append(mpatches.Patch(label=classes[i],color=colors[i]))
    ax.legend(handles=handles)
    for i in range(n_trees):
        ax = plt.subplot(gs[2,i])
        a = trees[i] + 1
        a[0] =0
        McNeuron.visualize.plot_dendrogram(a, pass_ax=True, ax=ax)
