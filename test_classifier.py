#! /usr/bin/env python3.6

from argparse import ArgumentParser
from classify_functions import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,\
    QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score


parser = ArgumentParser()
parser.add_argument('-t', type=str, help='Morphometric data')
parser.add_argument('--classifier', type=str, help='Classifier type')
# Valid values are: 'kNN', 'NB', 'SVM', 'DT', 'DF', 'LDA', 'QDA' or 'ensemble'
parser.add_argument('--training_set', type=float, default=0.75,
                    help='Relative size of training set')
parser.add_argument('--permutations', type=int, default='10',
                    help='Number of permutations of training/testing set')
parser.add_argument('-k', type=int, default=6,
                    help='k for k-NN. Ignored with other classifiers')
parser.add_argument('--kernel', type=str, default='rbf',
                    help='Kernel for SVM. Ignored by other classifiers')
parser.add_argument('--show_tree', action='store_true',
                    help='Print decision tree(s). Ignored by other classifiers')
parser.add_argument('--tree_num', type=int, default=10,
                    help='Number of trees for random forest. Ignored by other classifiers')
parser.add_argument('--fragilaria', action='store_true',
                    help='Tweaks specific to Fragilaria/Ulnaria dataset')
args = parser.parse_args()

if 1 <= args.training_set <= 100:
    training_set = args.training_set / 100
elif 0 < args.training_set < 1:
    training_set = args.training_set
else:
    raise ValueError('Training set size should be either between 0 and 1, or between 1 and 100')

if args.fragilaria:
    data, strains = morpho_data(args.t,
                                strain_field=1,
                                data_fields=(5, 6, 7, 8, 9, 10, 11, 12),
                                scale_data=False) #ndarray of data, list of clade IDs
    strain_to_clade = clade_dict()
    labels = [strain_to_clade[x] for x in strains]
    is_claded = [bool(x) for x in labels]
    #Cells for which the clade assignment is known
    claded_data = data[is_claded]
    claded_strains = bool_indexing(strains, is_claded)
    claded_labels = [strain_to_clade[x] for x in claded_strains]
    # Cells for which the clade assignment is absent
    is_uncladed = [not x for x in is_claded]
    uncladed_data = data[is_uncladed]
    uncladed_strains = bool_indexing(strains, is_uncladed)
else:
    data, labels = morpho_data(args.t, strain_field=0, scale_data=False)

# Collecting accuracies for avg estimates
acc = []
info = []
test_consistency = []
uncladed_consistency = []
for _ in range(args.permutations):
    if args.fragilaria:
        # Per-strain training/testing, specific to fragilaria dataset
        training_data, training_strains,\
        testing_data, testing_strains = training_and_testing(claded_data, claded_strains, 0.75)
        training_labels = [strain_to_clade[x] for x in training_strains]
        testing_labels = [strain_to_clade[x] for x in testing_strains]
        tmp = sorted(set(training_strains))
    else:
        training_data, testing_data,\
        training_labels, testing_labels = train_test_split(data, labels,
                                                train_size=args.training_set)
    if args.classifier == 'kNN':
        classifier = KNeighborsClassifier(n_neighbors=args.k, metric='cosine')
        big_classifier = KNeighborsClassifier(n_neighbors=args.k,
                                              metric='cosine')
    elif args.classifier == 'NB':
        # Only tested Gaussian because we can kinda sorta assume morphology of
        # specimen to be distributed approx. normally
        classifier = GaussianNB()
        big_classifier = GaussianNB()
    elif args.classifier == 'SVM':
        classifier = SVC(kernel=args.kernel)
        big_classifier = GaussianNB()
    elif args.classifier == 'DT':
        classifier = DecisionTreeClassifier()
        big_classifier = DecisionTreeClassifier()
    elif args.classifier == 'DF':
        classifier = RandomForestClassifier(n_estimators=args.tree_num)
        big_classifier = RandomForestClassifier(n_estimators=args.tree_num)
    elif args.classifier == 'LDA':
        classifier = LinearDiscriminantAnalysis(solver='lsqr')
        big_classifier = LinearDiscriminantAnalysis(solver='lsqr')
    elif args.classifier == 'QDA':
        classifier = QuadraticDiscriminantAnalysis()
        big_classifier = QuadraticDiscriminantAnalysis()
    elif args.classifier == 'ensemble':
        classifier = VotingClassifier(estimators=[
            ('NB', GaussianNB()),
            ('SVM_l', SVC(kernel='linear')),
            #('SVM_rbf', SVC(kernel='rbf')),
            ('SVM_poly', SVC(kernel='poly')),
            ('kNN', KNeighborsClassifier(n_neighbors=6, metric='cosine')),
            ('DT', DecisionTreeClassifier()),
            ('DF', RandomForestClassifier(n_estimators=100)),
            ('LDA', LinearDiscriminantAnalysis()),
            ('QDA', QuadraticDiscriminantAnalysis())
            ], n_jobs=3)
        big_classifier = VotingClassifier(estimators=[
            ('NB', GaussianNB()),
            ('SVM_l', SVC(kernel='linear')),
            #('SVM_rbf', SVC(kernel='rbf')),
            ('kNN', KNeighborsClassifier(n_neighbors=6, metric='cosine')),
            ('DT', DecisionTreeClassifier()),
            ('DF', RandomForestClassifier(n_estimators=100)),
            ('LDA', LinearDiscriminantAnalysis())
        ], n_jobs=3)
    classifier.fit(training_data, training_labels)
    classified_labels = classifier.predict(testing_data)
    # Estimating accuracy and mutual info on training/testing sets
    acc.append(
        accuracy_score(testing_labels, classified_labels, normalize=True))
    info.append(adjusted_mutual_info_score(testing_labels, classified_labels))
    if args.fragilaria:
        # Testing intrastrain consistency on testing set
        test_consistency.append(
            intrastrain_consistency(testing_strains, classified_labels))
print(f'Average accuracy {sum(acc) / len(acc):.2f}, average adjusted mutual info {sum(info) / len(info):.2f}')
if args.fragilaria:
    # Intrastrain consistency on non-claded data
    # This doesn't need to be done on every permutation because the whole dataset is
    # used for training anyway
    big_classifier.fit(claded_data, claded_labels)
    if args.classifier == 'DT' and args.show_tree:
        print(export_text(big_classifier, ['Length (μm)', 'Width proximally (μm)', 'Stria per 10 μm', 'Areolae in stria', 'Rimoportula', 'Width distally at apices (μm)','Width distally below the apices (μm)', 'Areolae in 10 μm']))
    uncladed_labels = big_classifier.predict(uncladed_data)
    uncladed_consistency.append(intrastrain_consistency(uncladed_strains,
                                                        uncladed_labels))
    print(f'Intrastrain consistency {sum(test_consistency)/len(test_consistency):.2f} on test data, {sum(uncladed_consistency)/len(uncladed_consistency):.2f} on unsequenced strains\n')