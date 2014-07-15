"""
File: test_classifiers.py

Author: Robert Elwell and Isaac Caswell
Created: [Unknown, but modifications by Isaac starting July 1]

Usage:
  i. command line:
  $  python test_classifiers.py [--class-file class_file] [--features-file features_file] [--verbose]
  
  ii. from within a script:
  import test_classifiers as tc
  tc.get_classifier_accuracies(features_file [,class_file [, verbose]])


Functionality: Tests a variety of classifiers on wikia data, using leave out one cross validation, and returns their 
accuracies, runtimes, and individual predictions (in case the client wants to experiment with ensemble classifiers).
class-file is a mapping of wikia IDs to their hand labeled class.
feature-file is a mapping of (lots of) wikia ids to word features, automatically generated via extract_wiki_data.py 

Documentation by Isaac, so there is room for error in my interpretation.
"""



import traceback
import time
from  __init__ import vertical_labels, Classifiers
#from . import vertical_labels, Classifiers
from collections import OrderedDict, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
from argparse import ArgumentParser, FileType
import numpy as np
from sklearn.linear_model import RandomizedLogisticRegression


# constants:

# USE_TOP_N_FEATURES: the amount of features to keep after feature selection
USE_TOP_N_FEATURES = 1000 

def get_args():
    ap = ArgumentParser()
    ap.add_argument('--class-file', type=FileType('r'), dest='class_file')
    ap.add_argument('--features-file', type=FileType('r'), dest='features_file')
    ap.add_argument('--verbose', dest = 'verbose', action = 'store_true', default = False)
    return ap.parse_args()


def main():
    """
    Note: functionality has been exported to get_classifier_accuracies. --Isaac
    """
    args = get_args()
    print get_classifier_accuracies(args.features_file, args.class_file, args.verbose)


def select_most_important_features(training_data, actual_labels, N_features = USE_TOP_N_FEATURES): # Added by Isaac
    """
    performs feature selection on the given data and returns the data as a matrix pruned to the top N_features.
    :param feature_rows: data matrix for training samples. shape = [n_samples, n_features]
    :type feature_rows: array-like
    :param actual_labels: a list where actual_labels[i] is the class (vertical label) of data instance i
    :type actual_labels: list
    :param N_features: how many features to select for
    :type N_features: int
    """
    actual_labels = np.array(actual_labels)
    print actual_labels
    # training_data = np.array(training_data)
    print training_data.nonzero()

    randomized_logistic = RandomizedLogisticRegression()
    if N_features:
        randomized_logistic.fit(training_data, actual_labels) 
        top_N_features = np.argsort(randomized_logistic.scores_)[0:N_features]
        lrther
        return training_data[:, top_N_features]
    else:
        return randomized_logistic.fit_transform(training_data, actual_labels)


def get_classifier_accuracies(features_file, class_file = None, verbose = False):
    """same functionality as main(), but easier to call from wrapper script.   This function 
    to be called in test_folder_of_data.py
    created by Isaac
    """

    if class_file:
        groups = defaultdict(list)
        first_line = 1 #first line labels columns and is tis unnecessary.  changed by Isaac
        for line in class_file: #changed by Isaac
            if first_line:
                first_line = 0
                continue
            splt = line.strip().split(',')
            if splt[0] != '': # added by Isaac
                groups[splt[1]].append(int(splt[0]))
    else:
        groups = vertical_labels

    if verbose: # --Isaac
        print u"Loading CSV..."
    """
    lines = [line.decode(u'utf8').strip().split(u',') for line in features_file]
    # lines = [splt for splt in lines if splt[0]!='']# This added to original version.  Removes the lines with an entry only from the "Secondaries to choose from" column in the spreadsheet --Isaac
    # lines = lines[1:]#changed by Isaac: first line is the labels of the columns ('wikia_id, ....'), so I remove it # Edit: only good if we're using secondary labels as features
    lines = [splt for splt in lines if int(splt[0]) in [v for g in groups.values() for v in g]] # i.e. take only lines that are in the coded testing data sat
    wid_to_features = OrderedDict([(splt[0], u" ".join(splt[1:])) for splt in lines])
    """

    wid_to_features = OrderedDict([(splt[0], u" ".join(splt[1:])) for splt in
                                   [line.decode(u'utf8').strip().split(u',') for line in features_file]
                                   if int(splt[0]) in [v for g in groups.values() for v in g]  # only in group for now
                                   ])
    

    if verbose: # --Isaac
        print u"Vectorizing..."
    vectorizer = TfidfVectorizer()
    data = [(str(wid), i) for i, (key, wids) in enumerate(groups.items()) for wid in wids]
    wid_to_class = dict(data)
    feature_keys = wid_to_features.keys()
    feature_rows = wid_to_features.values()
    vectorizer.fit_transform(feature_rows)
    #Note: using vectorizer.transform() so often may be a wee bit inefficient.... 
    feature_rows = select_most_important_features(vectorizer.transform(feature_rows), [wid_to_class[str(wid)] for wid in feature_keys]) # Added by Isaac
    vectorizer.fit_transform(feature_rows) # Refit is now that ... uh... TODO figure this sh8 out

    loo_args = []

    if verbose: # --Isaac
        print u"Prepping leave-one-out data set..."
    for i in range(0, len(feature_rows)):
        feature_keys_loo = [k for k in feature_keys]
        feature_rows_loo = [f for f in feature_rows]
        loo_row = feature_rows[i]
        loo_class = wid_to_class[str(feature_keys[i])]
        del feature_rows_loo[i]
        del feature_keys_loo[i]
        loo_args.append(
            (vectorizer.transform(feature_rows),                # train
             [wid_to_class[str(wid)] for wid in feature_keys],  # classes for training set
             vectorizer.transform([loo_row]),                   # predict  # The features corresponding to the instance whose label is to be predicted --Isaac
             [loo_class]                                        # expected class
             )
        )

    print u"Running leave-one-out cross-validation..."

    p = Pool(processes=8)
    return p.map_async(classify, [((name, clf), loo_args, verbose) for name, clf in Classifiers.each_with_name()]).get()


def classify(arg_tup):
    start = time.time()
    try:
        (name, clf), loo, verbose = arg_tup # Added verbose --Isaac
        predictions = []
        expectations = []
        prediction_probabilities = [] # where prediction_probabilities[i][j] is the probability that instance i is of class j --Isaac
        for i, (training, classes, predict, expected) in enumerate(loo):
            # predict is the feature vectore corresponding to a single data instance, i.
            if verbose: # --Isaac
                print name, i
            clf.fit(training.toarray(), classes)
            prediction_probabilities.append(clf.predict_proba(predict.toarray())) # --Isaac (for ensemble classifier)
            predictions.append(clf.predict(predict.toarray()))
            #RODO: Try: (should be same, but isn't always. unsure why.)  
            # predictons.append([np.argmax(prediction_probabilities[i])])
            #if predictions[i] != np.argmax(prediction_probabilities[i]):
            #    print "hoho!: \n%s  \n\t%s\n\t%s"%(prediction_probabilities[i], predictions[i], np.argmax(prediction_probabilities[i]))
                                               
            expectations.append(expected)
        correct_vec = [predicted == actual for predicted, actual in zip(predictions, expectations)] # decomposition added by Isaac 
        score = sum(correct_vec) # result of decomp
        # score = len([i for i in range(0, len(predictions)) if predictions[i] == expectations[i]]) # old version
        score = score*1.0/len(expectations) #score modified by Isaac: I thought percentages were more meaningful
        if verbose: # --Isaac
            print name, score, time.time() - start
        # Note on return statement: predictions is returned to enable a zero redundancy ensemble classifier
        # (i.e. classifications need not be recalculated) for clients of this module.
        # expectations is returned to evaluate this predictor.  For most implementations, expectations will be very redundant, 
        # an identical copy being returned for each classifier within a single data instance.  (i.e. 10x too many times).
        # This isn't the bottleneck, though, so I'll leave it be.
        return name, score, time.time() - start, prediction_probabilities, expectations
    except Exception as e:
            print e
            print traceback.format_exc()


if __name__ == u'__main__':
    main()
