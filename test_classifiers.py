"""
File: test_classifiers.py

Author: Robert Elwell and Isaac Caswell
Created: [Unknown, but modifications by Isaac starting July 1]

USAGE:
  i. command line:
  $  python test_classifiers.py [--class-file class_file] [--features-file features_file] [--verbose] [--feature-select 5000] [--sample-size 100]
  
  ii. from within a script:
  import test_classifiers as tc
  tc.get_classifier_accuracies(features_file [,class_file [, verbose [,return_ensemble_materials[, feature_select[, sample_size]]]]])


FUNCTIONALITY: Tests a variety of classifiers on wikia data, using leave out one cross validation, and returns their 
accuracies, runtimes, and individual predictions (in case the client wants to experiment with ensemble classifiers).
if USE_TOP_N_FEATURES, preliminary feature selection is performed on the data prior to running the classifiers on them.
The feature selection (random logistic regression) will make the program run faster, avoid overfitting, and be easier
to interpret, if you want to go in and look at the features to see what's important.


PARAMETERS:
class-file is a mapping of wikia IDs to their hand labeled class.
feature-file is a mapping of (lots of) wikia ids to word features, automatically generated via extract_wiki_data.py 
feature-select: 0 if no feature-selection to be done.  Otherwise, the amount of features to leave after feature selection. 

BUGS/PROBLEMS:
  -always gives "AttributeError: log" error, although this does not appear to hamper performance
  -often (always?) gives "SVD did not converge" error
  -feature selection appears not to work.  source not certain.
TODO:
  -incorporate in argument for extra_preprocessed_features.  This file would contain features such as page views, 
  which can be added into the dataset without the text processing that is done here.

Documentation by Isaac, so there is room for error in my interpretation of Robert's parts.
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
import random
# constants:

# USE_TOP_N_FEATURES: the amount of features to keep after feature selection.  If this is 0, no feature selection is performed.
USE_TOP_N_FEATURES = 1000 

# RETURN_ENSEMBLE_MATERIALS:
RETURN_ENSEMBLE_MATERIALS = False

# PRUNE_SET_FOR_TESTING: make the dataset into a manageable size for testing.  If it is 0, no pruning occurs.
# Otherwise, dataset is pruned to this many training instances.
PRUNE_SET_FOR_TESTING = 100

def get_args():
    ap = ArgumentParser()
    ap.add_argument('--class-file', type=FileType('r'), dest='class_file')
    ap.add_argument('--features-file', type=FileType('r'), dest='features_file')
    ap.add_argument('--verbose', dest = 'verbose', type=int, default = 1)
    ap.add_argument('--feature-select', dest = 'feature_select', type=int, default = USE_TOP_N_FEATURES)
    ap.add_argument('--sample-size', dest = 'sample_size', type=int, default = PRUNE_SET_FOR_TESTING)
    return ap.parse_args()


def main():
    """
    Note: functionality has been exported to get_classifier_accuracies. --Isaac
    """
    args = get_args()
    print get_classifier_accuracies(args.features_file, args.class_file, args.verbose, return_ensemble_materials = False, feature_select=args.feature_select, sample_size = args.sample_size)

def verbose_print(msg, program_verbosity_level, msg_verbosity_level=1):
    """
    usage:
    verbose_print("message", verbose)
    verbose_print("really detailed message", verbose, 2)
    verbose_print("you really want me to say everything I'm doing, dontcha?", 3)
    """
    if program_verbosity_level >=msg_verbosity_level:
        print msg


def select_most_important_features(training_data, actual_labels, N_features, penalty = 'random'): # function Added by Isaac
    """
    performs feature selection on the given data and returns an array of the indices of the top N_features.
    Reasons for use: 
       -dimensionality reduction equals time of calculation reduction, and otherwise this might take a day per language to learn
       -reduce overfitting
       -possibility for increased interpretability of data
    
    :param feature_rows: data matrix for training samples. shape = [n_samples, n_features]
    :type feature_rows: array-like
    :param actual_labels: a list where actual_labels[i] is the class (vertical label) of data instance i
    :type actual_labels: list
    :param N_features: how many features to select for
    :type N_features: int
    :return: [0]: array of the top most important features. [1]: the weights of those features
    :rtype: tuple
    
    """
    actual_labels = np.array(actual_labels)

    randomized_logistic = RandomizedLogisticRegression()
    
    randomized_logistic.fit(training_data, actual_labels) 
    #top_N_features = np.argsort(randomized_logistic.scores_)[0:N_features]
    top_N_features = np.argsort(randomized_logistic.scores_)[-N_features:] #caught this on 22 July, 4:25pm.  whoops.
    return (top_N_features)





def do_feature_selection(vectorizer, feature_rows, actual_labels, verbose, N_features):
    """
    see documentation in select_most_important_features
    """
    if not N_features:
        verbose_print("No feature selection", verbose, 2)
    else:
        verbose_print("Performing feature selection....", verbose)
        feats_to_keep_id = select_most_important_features(vectorizer.transform(feature_rows), actual_labels, N_features = N_features)
        feats_to_keep_name = np.array(vectorizer.get_feature_names())[feats_to_keep_id]
        pp = vectorizer.build_analyzer()
        #pp = vectorizer.build_preprocessor() # if this used, use pp(row).split(" ") in comprehension below

        feature_rows = [" ".join([feat for feat in pp(row) if feat in feats_to_keep_name]) for row in feature_rows]

        vectorizer = TfidfVectorizer() # Refit now that we have a new feature set
        vectorizer.fit_transform(feature_rows)
        verbose_print("Pared down to %s features!"%N_features, verbose)
    return vectorizer, feature_rows

def add_numeric_features(X_train, wid_to_numeric_features):
    """
    NOTE: THIS function only works if tfidf_vectorizer preserves the order of its argument, feature_rows.  
    Otherwise we are doomed and there is no way (except a really silly hacky way I'll describe later maybe)
    of adding in numeric features.
    """
    N_numeric_feats = len(wid_to_numeric_features[wid_to_numeric_features.keys()[0]])
    augmented_shape = (X_train.shape[0], X_train.shape[1] + N_numeric_feats)
    X_train._shape = (augmented_shape)
    for i, wid in enumerate(wid_to_numeric_features):
        feats = wid_to_numeric_features[wid]
        X_train[i, -N_numeric_feats:] = feats

    return X_train

def get_classifier_accuracies(features_file, numeric_features_file = None, class_file = None, verbose = 0, return_ensemble_materials = False, \
                                  feature_select = USE_TOP_N_FEATURES, sample_size = PRUNE_SET_FOR_TESTING, \
                                  return_training_error = False, use_numeric_features=True):
    """same functionality as main(), but easier to call from wrapper script.   This function
    to be called in test_folder_of_data.py
    

    If you just want accuracy and time of execution, call with return_ensemble_materials=false.
    If you want the predictions and labels for each data instance oer classifier, call with return_ensemble_materials=true.
    created by Isaac
    
    :param class_file: a mapping of wikia IDs to their hand labeled class.
    :type class_file: file ('filestream'? 'file object?' I'm bad with terminology.)
    :param features_file: a mapping of (lots of) wikia ids to word features, automatically generated via extract_wiki_data.py
      Note: each line of features_file is of the form "1234,spiderman,silk,hand" etc. 
    :type features_file: csv file
    :param numeric_features_file: a mapping of wikia ids to numeric (reaor integer valued) features, automatically generated via extract_wiki_data.py
      Note: each line of features_file is of the form "1234,12,32,99,123.5" etc. 
    :type numeric_features_file: csv file
    :param verbose: should the function say useless yet comforting things as it runs?
    :type verbose: bool
    :param return_ensemble_materials: should the return value include predictions and labels for each data instance per classifier?
                                      Set to True only if you're doing ensemble testing (i.e. test_folder_of_data.py)
                                      (For documentation on how to use for ensemble classification, see test_folder_of_data.py)
                                      If you just want accuracy and speed of classifiers, set to false.
    :type return_ensemble_materials: bool
    """


    verbose_print("%s numeric (preprocessed) features...."%("Including" if use_numeric_features else "OMITTING"), verbose, 1)

    global RETURN_ENSEMBLE_MATERIALS
    RETURN_ENSEMBLE_MATERIALS = return_ensemble_materials # TODO: fix this nonsense
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
        print "Whoops you done messed up real good.  You don't never want to be forgetting a class file."
        groups = vertical_labels

    verbose_print(u"Loading CSV...", verbose)


    # Note: after [v for g in groups.values() for v in g]  Robert wrote 'only in group for now',  Why did he say 'for now'?

    wid_to_features = OrderedDict([(splt[0], u" ".join(splt[1:])) for splt in
                                   [line.decode(u'utf8').strip().split(u',') for line in features_file]
                                   if int(splt[0]) in [v for g in groups.values() for v in g]
                                   ])



    wid_to_numeric_features = OrderedDict([(splt[0], [float(i) if i != 'None' else 0 for i in splt[1:]]) for splt in
                                   [line.strip().split(u',') for line in numeric_features_file]
                                   if int(splt[0]) in [v for g in groups.values() for v in g]
                                   ])
    assert wid_to_features.keys() == wid_to_numeric_features.keys()
    #NUMERIC FEATURE_TODO: actually do something with these.
    verbose_print(u"Dataset has %s data instances"%(len(wid_to_features.values())), verbose)
    if sample_size: #make the dataset into a manageable size for testing.
        doomed_keys = wid_to_features.keys()
        random.seed(9) # just for comparative testing porpoises
        random.shuffle(doomed_keys)
        doomed_keys = doomed_keys[sample_size:]
        for doomed_key in doomed_keys:
            del wid_to_features[doomed_key]
            del wid_to_numeric_features[doomed_key]
        verbose_print(u"Pruned dataset to %s instances"%(len(wid_to_features.keys())), verbose)


    verbose_print(u"Vectorizing...", verbose)
    data = [(str(wid), i) for i, (key, wids) in enumerate(groups.items()) for wid in wids]
    wid_to_class = dict(data)
    feature_keys = wid_to_features.keys()
    feature_rows = wid_to_features.values()
    vectorizer = TfidfVectorizer()
    
    # Note that the vectorizer splits tokens like "TOP_ART:zombi_brain" into two tokens.  For this reason I have changed the feature marking into TOP_ART_zombi_brain etc.
    vectorizer.fit_transform(feature_rows) # NB: as far as I can tell, it makes more sense to have vectorizer.fit(feature_rows), but I'll trust Robert here still...
    verbose_print("Dataset has %s features"%vectorizer.idf_.shape, verbose) #NOTE: !!! for some machines (computers), like aws I think, one has to write "vectorizer.tfidf.idf_.shape".  No idea why.
    loo_args = []

    verbose_print(u"Prepping leave-one-out data set...", verbose)
    for i in range(0, len(feature_rows)):
        #integrate other features here, like pagecount.
        feature_keys_loo = [k for k in feature_keys]
        feature_rows_loo = [f for f in feature_rows]
        loo_row = feature_rows[i]
        loo_class = wid_to_class[str(feature_keys[i])]
        del feature_rows_loo[i]
        del feature_keys_loo[i]
        vectorizer_loo = vectorizer
        train_labels_loo = [wid_to_class[str(wid)] for wid in feature_keys_loo] # Isaac

        if loo_class not in train_labels_loo:
            print "Unique element with class %s. Skipping this cross validation cross section."%loo_class
            continue
        if feature_select: # Added by Isaac. 
            vectorizer_loo, feature_rows_loo = do_feature_selection(vectorizer, feature_rows_loo, \
                                                                   actual_labels = [wid_to_class[str(wid)] for wid in feature_keys_loo], verbose = verbose, N_features = feature_select)



        X_train = vectorizer_loo.transform(feature_rows_loo)
        predict = vectorizer_loo.transform([loo_row])             # changed to vectorizer_loo  --Isaac        
        if use_numeric_features:
            wid_loo = wid_to_numeric_features.keys()[i]        
            wid_to_numeric_features_loo = wid_to_numeric_features.copy()
            del wid_to_numeric_features_loo[wid_loo]        
            X_train = add_numeric_features(X_train, wid_to_numeric_features_loo)
            predict = add_numeric_features(predict, OrderedDict([(wid_loo, wid_to_numeric_features[wid_loo])]))

        
        loo_args.append(
            (X_train,                                               # train  #[source of erstwhile LOOCV bug]
             train_labels_loo,                                      # classes for training set #[source of erstwhile LOOCV bug]
             predict,                                               # predict  # The features corresponding to the instance whose label is to be predicted --Isaac
             [loo_class]                                            # expected class
             )
        )



    print u"Running leave-one-out cross-validation..."

    p = Pool(processes=8)
    #QDA always gave errors and crashed. Linear SVM strted hanging seemingly infinitely even on 12 data instances, and was never good anyways.
    all_clf = [u"QDA", u"Linear SVM", u"Maximum Entropy", u"LDA", u"Naive Bayes", u"AdaBoost", u"Random_Forest", u"Decision Tree", u"RBF_SVM", u"Nearest Neighbors"] # For reference
    omit_clf = [u"QDA", u"Linear SVM", u"LDA", u"RBF_SVM", "AdaBoost"] # LDA was never good, and took forever.
    #omit_clf = list(set(all_clf).difference(set([u"Random_Forest"])))
    print omit_clf
    return_value = p.map_async(classify, [((name, clf), loo_args, verbose) for name, clf in Classifiers.each_with_name() if name not in omit_clf]).get()
    if return_training_error:
        verbose_print("Getting Training Error......", verbose, 1)
        X = vectorizer.transform(feature_rows)
        y = [wid_to_class[str(wid)] for wid in feature_keys]
        training_error = p.map_async(get_training_error, [((name, clf), X, y, verbose) for name, clf in Classifiers.each_with_name() if name not in omit_clf]).get()
        return_value = (return_value, training_error)
    
    return return_value



def classify(arg_tup):
    start = time.time()
    try:
        (name, clf), loo, verbose = arg_tup # Added verbose --Isaac
        predictions = []
        expectations = []
        prediction_probabilities = [] # where prediction_probabilities[i][j] is the probability that instance i is of class j --Isaac
        for i, (training, classes, predict, expected) in enumerate(loo):
            # predict is the feature vectore corresponding to a single data instance, i.
            verbose_print("%s predicting data instance %s..."%(name, i), verbose, 2)
            clf.fit(training.toarray(), classes)
            if RETURN_ENSEMBLE_MATERIALS:
                prediction_probabilities.append(np.squeeze(clf.predict_proba(predict.toarray()))) # --Isaac (for ensemble classifier)
            predictions.append(clf.predict(predict.toarray()))
            #TODO: Try: (should be same, but isn't always. unsure why.)  
            # predictons.append([np.argmax(prediction_probabilities[i])])
            #if predictions[i] != np.argmax(prediction_probabilities[i]):
            #    print "hoho!: \n%s  \n\t%s\n\t%s"%(prediction_probabilities[i], predictions[i], np.argmax(prediction_probabilities[i]))
                                               
            expectations.append(expected)
        correct_vec = [predicted == actual for predicted, actual in zip(predictions, expectations)] # decomposition added by Isaac 
        score = sum(correct_vec) # result of decomp
        # score = len([i for i in range(0, len(predictions)) if predictions[i] == expectations[i]]) # old version
        score = score*1.0/len(expectations) #score modified by Isaac: I thought percentages were more meaningful
        verbose_print("%s: \n\taccuracy:%s\n\ttime:%s"%(name, score, time.time() - start), verbose, 2)
        # Note on return statement: predictions is returned to enable a zero redundancy ensemble classifier
        # (i.e. classifications need not be recalculated) for clients of this module.
        # expectations is returned to evaluate this predictor.  For most implementations, expectations will be very redundant, 
        # an identical copy being returned for each classifier within a single data instance.  (i.e. 10x too many times).
        # This isn't the bottleneck, though, so I'll leave it be.
        if RETURN_ENSEMBLE_MATERIALS:
            return name, score, time.time() - start, prediction_probabilities, expectations
        else: 
            return name, score, time.time() - start
    except Exception as e:
            print e
            print traceback.format_exc()




def get_training_error(arg_tup):
    """
    This try/except structure is useful when testing for instance with QDA, which always gave errors and crashed.
    NOTE: right now actually returns the training accuracy, rather than error, because it seemed more useful.
    """
    try:
        (name, clf), X, y, verbose = arg_tup # Added verbose --Isaac
        N = len(y)
        predictions = []
        verbose_print("Predicting training error for %s..."%(name), verbose, 2)
        clf.fit(X.toarray(), y)
        predictions = clf.predict(X.toarray())
        
        correct_vec = [predicted == actual for predicted, actual in zip(predictions, y)] # decomposition added by Isaac 
        # error = 1.0 - sum(correct_vec)*1.0/N # result of decomp
        error = sum(correct_vec)*1.0/N # result of decomp
        
        verbose_print("%s: \n\ttraining accuracy:%s"%(name, error), verbose, 2)
        # See note on return statement for classify().
        prediction_probabilities = clf.predict_proba(X.toarray())

        if RETURN_ENSEMBLE_MATERIALS:
            return name, error, list(prediction_probabilities), y
        else:
            return name, error
    except Exception as e:
        print e
        print traceback.format_exc()





if __name__ == u'__main__':
    main()



"""
appendix: deleted code which I might regret deleting sometime
    lines = [line.decode(u'utf8').strip().split(u',') for line in features_file]
    # lines = [splt for splt in lines if splt[0]!='']# This added to original version.  Removes the lines with an entry only from the "Secondaries to choose from" column in the spreadsheet --Isaac
    # lines = lines[1:]#changed by Isaac: first line is the labels of the columns ('wikia_id, ....'), so I remove it # Edit: only good if we're using secondary labels as features
    lines = [splt for splt in lines if int(splt[0]) in [v for g in groups.values() for v in g]] # i.e. take only lines that are in the coded testing data sat
    wid_to_features = OrderedDict([(splt[0], u" ".join(splt[1:])) for splt in lines])


L1 penalization for feature selection: (discontinued because it returns matrix not list of IDS, which was needed to prune out feature_rows, which is a list of lists, not a matrix.)

    actual_labels = np.array(actual_labels)
    selector = None
    if penalty == 'random':
        selector = RandomizedLogisticRegression()
    elif penalty == 'l1' or penalty == 'l2':
        selector = LogisticRegression(penalty = penalty)
    ...

"""
