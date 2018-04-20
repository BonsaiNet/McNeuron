"""Distance utility"""
import numpy as np
#np.random.seed(100)

def get_feature_neuron(neuron,
                       hist_features,
                       value_features,
                       vec_value):
    """
    Getting the mean of neuron.

    Parameters:
    -----------
    neuron: Neuron
    hist_features: dic
        the dictionary of feature with the bins of histogram
    value_features: list
        the list of features

    Return:
    -------
    mean_hist: dic
        the dictionary of features. The value of each feature is
        a numpy array of mean of neuron's feature at the given bin of
        histogram.
    mean_value: dic
        the dictionary of features. The value of each feature is
        a scalar feature of the neuron. In the case that the feature
        is a vector it return the mean of vector.
    """
    mean_hist = {}
    mean_value = {}
    mean_vec_value = {}
    for name in hist_features.keys():
        a = neuron.features[name]
        a = a[~np.isnan(a)]
        hist_range = hist_features[name]
        a = np.histogram(a, bins=hist_range)[0].astype(float)
        if(a.sum() != 0):
            a = a/a.sum()
        else:
            a = np.zeros(len(hist_range)-1)
        mean_hist[name] = a

    for name in value_features:
        a = neuron.features[name]
        mean_value[name] = a.mean()

    for name in vec_value:
        a = neuron.features[name]
        mean_vec_value[name] = a

    return mean_hist, mean_value, mean_vec_value

def distance_from_database(neuron, database):
    return distance(neuron,
                    verbose=0,
                    list_hist_features=database.hist_features.keys(),
                    hist_range=database.hist_features,
                    mean_hist=database.mean_hist,
                    std_hist=database.std_hist,
                    list_value_features=database.mean_value.keys(),
                    mean_value=database.mean_value,
                    std_value=database.std_value,
                    list_vec_value=database.mean_vec_value.keys(),
                    std_vec_value=database.std_vec_value,
                    mean_vec_value=database.mean_vec_value)


def distance_from_database_with_name(neuron, database):

    list_hist_features = database.hist_features.keys()
    hist_range = database.hist_features
    mean_hist = database.mean_hist
    std_hist = database.std_hist
    list_value_features = database.mean_value.keys()
    mean_value = database.mean_value
    std_value = database.std_value
    list_vec_value = database.mean_vec_value.keys()
    std_vec_value = database.std_vec_value
    mean_vec_value = database.mean_vec_value

    distance = {}
    for k in range(len(list_hist_features)):
        name = list_hist_features[k]
        error, error_normal = l2_distance(neuron=neuron,
                                          name=name,
                                          range_histogram=hist_range[name],
                                          mean=mean_hist[name],
                                          std=std_hist[name])
        distance[name] = error


    k = 0
    for name in list_value_features:
        error, error_normal = gaussian_distance(neuron=neuron,
                                                name=name,
                                                mean=mean_value[name],
                                                std=std_value[name])
        distance[name] = error
    k = 0
    for name in list_vec_value:
        error, error_normal = gaussian_distance_vec(neuron=neuron,
                                                    name=name,
                                                    mean=mean_vec_value [name],
                                                    std=std_vec_value[name])
        distance[name] = error

    return distance



def distance(neuron,
             verbose,
             list_hist_features,
             hist_range,
             mean_hist,
             std_hist,
             list_value_features,
             mean_value,
             std_value,
             list_vec_value,
             std_vec_value,
             mean_vec_value):
    """
    Calculate log of probability distribution of neuron.

    Parameters:
    -----------
    neuron: Neuron
    """
    n_hist_features = len(list_hist_features)
    n_value_features = len(list_value_features)
    n_vec_value = len(list_vec_value)

    hist_error = np.zeros(n_hist_features)
    hist_error_normal = np.zeros(n_hist_features)

    value_error = np.zeros(n_value_features)
    value_error_normal = np.zeros(n_value_features)

    vec_value_error = np.zeros(n_vec_value)
    vec_value_error_normal = np.zeros(n_vec_value)

    k = 0
    for name in list_hist_features:
        error, error_normal = l2_distance(neuron=neuron,
                                          name=name,
                                          range_histogram=hist_range[name],
                                          mean=mean_hist[name],
                                          std=std_hist[name])
        hist_error[k] = error
        hist_error_normal[k] = error_normal
        k += 1

        if(verbose >= 1):
            print(name + ' : %s' % error)
    k = 0
    for name in list_value_features:
        error, error_normal = gaussian_distance(neuron=neuron,
                                                name=name,
                                                mean=mean_value[name],
                                                std=std_value[name])
        value_error[k] = error
        value_error_normal[k] = error_normal
        k += 1

        if(verbose >= 1):
            print(name + ' : %s' % error)
    k = 0
    for name in list_vec_value:
        error, error_normal = gaussian_distance_vec(neuron=neuron,
                                                    name=name,
                                                    mean=mean_vec_value [name],
                                                    std=std_vec_value[name])
        vec_value_error[k] = error
        vec_value_error_normal[k] = error_normal
        k += 1

        if(verbose >= 1):
            print(name + ' : %s' % error)

    total_error = hist_error.sum() + value_error.sum() + vec_value_error.sum()
    error = np.append(hist_error, value_error)
    error = np.append(error, vec_value_error)

    error_normal = np.append(hist_error_normal, value_error_normal)
    error_normal = np.append(error_normal, vec_value_error_normal)

    if(verbose >= 1):
        print('\nand its probability is: %s' % total_error)
    return total_error, error, error_normal


def gaussian_distance(neuron,
                      name,
                      std,
                      mean):
    feature = neuron.features[name]
    std_fea = (feature - mean)/std
    error_normal = std_fea.mean()
    error = np.sqrt((std_fea ** 2).mean())

    return error, error_normal


def gaussian_distance_vec(neuron,
                          name,
                          mean,
                          std):
    feature = neuron.features[name]
    diff_fea = (feature - mean)/std
    error = np.sqrt(((diff_fea ** 2)/std**2).mean())
    error_normal = (diff_fea/std).mean()
    return error, error_normal

def l2_distance(neuron,
                name,
                range_histogram,
                std,
                mean):
    """
    Parameters
    ----------
    neuron: Neuron

    """
    feature = neuron.features[name]
    feature = feature[~np.isnan(feature)]
    hist_fea = np.histogram(feature, bins=range_histogram)[0].astype(float)
    if(sum(hist_fea) != 0):
        hist_fea = hist_fea/sum(hist_fea)
    else:
        hist_fea = np.ones(len(hist_fea))/float(len(hist_fea))
    diff_fea = hist_fea - mean
    error = np.sqrt(((diff_fea ** 2)/std**2).mean())
    error_normal = (diff_fea/std).mean()
    return error, error_normal
