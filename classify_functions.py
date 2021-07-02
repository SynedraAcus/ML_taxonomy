"""
A number of useful functions and data for the classifier
"""
from collections import defaultdict
from sklearn.preprocessing import scale
import numpy as np
import random

clades_tmp = {'A': ('0.0228-YH Hybrid', '3B355', '5.0903-F', '3B357',  # U. acus, green
                        '0.0228-B Hybrid', '5.1015-C', '5.0227-E', 'MM117', 'BK497',
                        'AxBK280', 'MM98', '0.0319-YB Hybrid', 'BK495', 'AxA6'),
                  'B': ('BZ1', 'BZ3', 'MM121', 'BZ251', 'L252', 'BZ264', 'CHZ442',  # U.danica, blue
                        'LAB55', 'LAB59', 'LAB259'),
                  'C': ('MM99', 'MM101', 'MM103', 'MM106', 'ACH111', 'MM113',  # F.radians, red
                        'MM144', 'BK146', 'L149', 'L150', 'MM239', 'MM240', 'MM243',
                        'MM245', 'MM246')}

def clade_dict():
    """
    Generate a dict of strain-to-clade mappings.
    All data are hardcoded
    :return:
    """
    clades = defaultdict(lambda: None)
    for clade in clades_tmp:
        for strain in clades_tmp[clade]:
            clades[strain] = clade
    return clades


def morpho_data(infile, strain_field=1,
                data_fields=None, scale_data=True):
    data = []
    for line in open(infile):
        if line[0] != '#':
            a = line.rstrip().split('\t')
            data.append(a)
    if not data_fields:
        # If data fields are not set explicitly, use all columns
        # after strain_field.
        data_fields = [x for x in range(strain_field + 1, len(data[0]))]
    data_array = np.asarray([[float(x[i]) for i in data_fields] for x in data])
    if scale_data:
        data_array = scale(data_array)
    strains = [x[strain_field] for x in data]
    return data_array, strains


def sample_strains(percentage):
    """
    Return a list of strain IDs randomly selected from each clade
    :param percentage: float. A percentage of clade that should be sampled
    :return:
    """
    r = []
    for clade in clades_tmp:
        to_select = int(len(clades_tmp[clade])*percentage)
        r += random.sample(clades_tmp[clade], to_select)
    return r


def indices(labels, strains_to_sample):
    """
    Given a list of strain IDs and a smaller list of strains to be extracted,
    produce a boolean array for filtering NumPy data
    :param data_array:
    :param strain_list:
    :param strains_to_sample:
    :return:
    """
    return [x in strains_to_sample for x in labels]


def bool_indexing(data, bool_list):
    """
    Reproduces NumPy's indexing by a boolean list
    :param data:
    :param bool_list:
    :return:
    """
    return [data[x] for x in range(len(data)) if bool_list[x]]


def training_and_testing(data, strains, percentage):
    """
    Split the data into a training set and a testing set
    :param data: 
    :param strains: 
    :param percentage: 
    :return: 
    """
    training_strains = sample_strains(percentage)
    training_indices = indices(strains, training_strains)
    training_data = data[training_indices]
    training_labels = bool_indexing(strains, training_indices)
    testing_indices = [not x for x in training_indices]
    testing_data = data[testing_indices]
    testing_labels = bool_indexing(strains, testing_indices)
    return training_data, training_labels, testing_data, testing_labels


def intrastrain_consistency(strains, labels):
    """
    Calculate intrastrain consistency
    :param strains:
    :param labels:
    :return:
    """
    running_counts = defaultdict(lambda: 0)
    running_strain = ''
    consistencies = []
    for i in range(len(strains)):
        if strains[i] == running_strain:
            running_counts[labels[i]] += 1
        else:
            if running_strain:
                consistencies.append(max(running_counts.values())/sum(running_counts.values()))
            running_counts = defaultdict(lambda: 0)
            running_counts[labels[i]] += 1
            running_strain = strains[i]
    return sum(consistencies)/len(consistencies)
