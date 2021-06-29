#! /usr/bin/env python3.6

from argparse import ArgumentParser
from collections import defaultdict, Counter
from itertools import groupby
from sklearn.neighbors import KNeighborsClassifier

from classify_functions import *

parser = ArgumentParser('Process M.B.\'s data')
parser.add_argument('-t', type=str, help='Training data')
parser.add_argument('-s', type=str, help='Data to process')
args = parser.parse_args()

# Loading reference data
data, strains = morpho_data(args.t,
                            use_fields=(5,6,7,8,9,10,11),
                            scale_data=False)
strain_to_clade = clade_dict()
labels = [strain_to_clade[x] for x in strains]
is_claded = [bool(x) for x in labels]
claded_data = data[is_claded]
claded_strains = bool_indexing(strains, is_claded)
claded_labels = [strain_to_clade[x] for x in claded_strains]
knn = KNeighborsClassifier(n_neighbors=6, metric='cosine')
knn.fit(claded_data, claded_labels)

# Loading sample data
data, plates = morpho_data(args.s,
                           use_fields=(2, 3, 4, 5, 6, 7, 8),
                           scale_data=False)
by_plate = defaultdict(list)
for i in range(len(plates)):
    by_plate[plates[i]].append(data[i])
species_names = {'A': 'U. acus', 'B': 'U. danica', 'C': 'F. radians'}
lengths = {}
for plate in by_plate:
    predicted_labels = knn.predict(by_plate[plate])
    print(f'Plate {plate}:')
    print(Counter((species_names[i] for i in predicted_labels)))
    lengths[plate] = defaultdict(list)
    for i in range(len(predicted_labels)):
        lengths[plate][predicted_labels[i]] = data[i][0]
