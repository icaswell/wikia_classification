"""
file: test_folder_of_data.py
Author: @Isaac Caswell
created: 8.7.2014
updated: 9.7.2014

Overview:
  Script to run test_classifiers.py on a folder of CSV files and output the results together.
  Will output statistics about the various classifiers, inc. avg. accuracy and consistency 
  (i.e. SD of results); results will be correlated with language/language family, as well,
  to test the hypothesis that different classifiers work better with different languages.

  Note that test_classifiers has been modified to include the function get_classifier_accuracies()
  in order to be callable and provide data in a  useful form

Usage: [commnd line]
  #[python extract_wiki_data_all_langs.py] deprecated
  [python make_features_from_CSV_folder --csv-folder CSV_datasets --outfolder corresponding_feature_data]
  python test_folder_of_data.py --csv-folder CSV_datasets --features-folder feats_folder_numeric_test --verbose 2 --feature-select 0 --sample-size 12 --doc-file doc_file --nonumeric

Arguments:
  -classfile_folder: name of a folder (in current directory) of CSV files (classfiles)

Known bugs/problems:
  -test_classifiers barfs up a lot of UserErrors and RuntimeErrors, namely that log encountered an 
  invalid value in divide
  -unfortunate name
"""
import sys
import time
import traceback
import os
import test_classifiers as test
#from argparse import ArgumentParser
import sys
from google_doc_csv_to_id_csv import transform
import numpy as np
import itertools as it
import scipy.misc as sc
#import extract_wiki_data_all_langs as ewdal
from argparse import ArgumentParser


LANGUAGES_TO_ITERATE_OVER = ['Chinese', 'German', 'English', 'Portuguese', 'Spanish', 'French', 'Russian', 'Polish', 'Japanese']
LANGUAGES_TO_ITERATE_OVER = ['English']#'Chinese']#['German', 'Chinese',            'Portuguese', 'Spanish', 'French', 'Russian', 'Polish', 'Japanese']
#Supported langs: languages I've gotten stemmers and tokenizers working for.
SUPPORTED_LANGS = ['Danish', 'Dutch', 'English', 'Finnish', 'French', 'German', 'Hungarian', 'Italian', 'Norwegian', 'Portuguese', 'Romanian', 'Russian', 'Spanish', 'Swedish', 'Chinese']
#LANGUAGES_TO_ITERATE_OVER = set(LANGUAGES_TO_ITERATE_OVER).intersection(set(SUPPORTED_LANGS))


ENSEMBLES_TO_DISPLAY = 15
#GET_DATASET == 1  #if the features are to be extracted by the script (extract_wiki_ids_all_langs) #can't do this because it's multithreaded
FEATURE_SELECT = 0
SAMPLE_SIZE = 10 #must be greater than 8 for a good time (there's a bug in sklearn)
# language_name_to_code: a temporary solution, maybe; but it's easy enough to hardcode for a small dataset.
# Thus far unused.
LANGUAGE_NAME_TO_CODE = {"German": "de",
                         "English": "en",
                         "Chinese": "zh",
                         "French": "fr",
                         "Japanese": "ja",
                         "Polish": "pl",
                         "Portuguese": "pt",
                         "Russian": "ru",
                         "Spanish": "es"
                         }
"""LANGUAGE_CODE_TO_NAME = {"de": "German",
                         "en": "English",
                         "zh": "Chinese",
                         "fr": "French",
                         "ja": "Japanese",
                         "pl": "Polish",
                         "pt": "Portuguese",
                         "ru": "Russian",
                         "es": "Spanish"
                         }
"""


def get_args():
    ap = ArgumentParser()
    ap.add_argument('--csv-folder', default = "wikia_classes", dest='csv_folder')
    ap.add_argument('--features-folder', default = "wiki_data_all", dest='features_folder')
    ap.add_argument('--doc-file', default = None, dest='doc_file')
    ap.add_argument('--verbose', dest = 'verbose', type = int, default = 1)
    ap.add_argument('--feature-select', dest = 'feature_select', type = int, default = 1000)
    ap.add_argument('--sample-size', dest = 'sample_size', type = int, default = 1000)
    ap.add_argument('--nonumeric', dest = 'use_numeric_features', action="store_false", default = True)
    return ap.parse_args()


def display_stats(results, dataset_language, clf_names, training_errors):
    """
    returns a string which describes in human readable way the performance of different classifiers.
    Returns a string rather than printing to the console for the purpose that one may also write to 
    a file, to keep a log of the results.

    Note: right now I'm actually reporting training accuracy, not training error.
    
    :param results: results[i][j] is, for language (dataset) i, a tuple of (clf name, LOOCV clf accuracy for that language)
    :type results: list of lists of tuples
    :param dataset_language: dataset_language[i] is the name of the language which the ith entry of results corresponds to
    :type dataset_language: list of strings
    :param clf_names: the names of each classifier, so clf_names[i] = results[X][i][0], I think.  Redundant, so?
    "type clf_names: list?
    :param training_errors: training_errors[i][j] is, for language i, a tuple of (classifier, training error for that classifier for that language, prediction over training set, actual labels of training set)
     Only the first two entries of each tuple are used by this function; the second two are used to calculate training error on the ensemble classifiers.
    :type training_errors: list of lists of tuples
    """
    #First, let's look at the best cassifiers, on average:

    return_string = ""
    try:
        N_clfs = len(training_errors) # Should be around 9

        accuracies = [[tup[1] for tup in dataset] for dataset in results]
        avg_acc = [np.mean(np.array(i)) for i in zip(*accuracies)]
        sd_acc = [np.std(np.array(i)) for i in zip(*accuracies)]

        training_errors_avg =  {}
        for language_data in training_errors: # get average training error 
            for clf_data in language_data:
                if clf_data[0] not in training_errors_avg:
                    training_errors_avg[clf_data[0]] = 0
                training_errors_avg[clf_data[0]] += clf_data[1]/N_clfs


        sorted_avg_acc_and_name = sorted(zip(avg_acc, clf_names, sd_acc), reverse=True)
        return_string += "Average performance, SD and training error: \n"
        for tup in sorted_avg_acc_and_name:
            return_string += u"\t%s %.1f%% \u00B1 %.1f"%(tup[1] + (17 - len(tup[1]))*" ", tup[0]*100, tup[2]*100) # where 17 is the length of the longest name, plus 1, for prettiness of formatting
            return_string += u"   TA:%.1f%%\n"%(training_errors_avg[tup[1]]*100)
            
            
        # Now, let's compare classifiers between different languages:
        #language: [best classifiers]
        return_string += "\nLineup per language: \n"
        lineup_per_language = [sorted(zip(acc, clf_names), reverse=True) for acc in accuracies] 
        for i in range(len(dataset_language)):
            return_string += "%s: "%dataset_language[i]
            for tup in lineup_per_language[i]:
                return_string += "%s (%.3f), "%(tup[1], tup[0][0])
            return_string += "\n"
            
    except Exception as e:
        return_string += str(e)
        return_string +="\n"
        return_string += str(traceback.format_exc())
    
    return return_string
    


def get_ensemble_accuracy(ensemble, all_prediction_probs, expected):
    """
    :param ensemble: a tuple of classifier indices
    :type ensemble: tuple
    :param all_prediction_probs:  all_prediction_probs[i][j][0][k] is the probability, using classifier i, 
    :param all_prediction_probs:  all_prediction_probs[i][j][k] is the probability, using classifier i, 
    that data instance j is in class k, as calculated with LOOCV in test_classifiers.py
    :type all_prediction_probs: list
    :param expected: length N list of actual labels of the data instance
    :type expected: list
    :return: percent accuracy of given ensemble over the data instance
    :rtype: float
    """
    N_classifiers = len(all_prediction_probs)
    N_instances = len(all_prediction_probs[0])
    N_classes = len(all_prediction_probs[0][0])
    ensemble_pred_probs_unsummed = np.array(all_prediction_probs)[np.array(ensemble)]
    #If this were in numpy to start with, this would be much easier...
    ensemble_pred_probs = np.zeros((N_instances, N_classes))
    for clf_preds in ensemble_pred_probs_unsummed:
        ensemble_pred_probs[:, :] += np.array(clf_preds)


    predictions = list(ensemble_pred_probs.argmax(axis=1))

    return sum([p==e[0] for p, e in zip(predictions, expected)])*1.0/len(expected)


def predict_all_ensembles_precalc(ensemble_scores, clf_prediction_probs, expected, ENSEMBLE_SIZES):
    """
    simulates a simplistic ensemble classifier, by argmaxing sums of individual classifier predictions.
    Could be extended by, for instance, predicting how much to weight each prediction based on a meta classifier.

    determine ensemble predictions of all subsets by summing predictions (yes, they needed)
    calculate accuracy by comparing to expectations
    return to be printed.  aggregate in another function.
    "precalc" refers to the fact that the predictions of each classifier have already been calculated.    
    
    :param ensemble_scores: keys are tuples of strings representing classifiers.  Values are sums of accuracies.
    :type ensemble_scores: dict
    :param clf_prediction_probs: predictions[i][j][k] is the probability that data instance j is of class k, according
     to classifier i.
    :type clf_prediction_probs: a length M list of length N lists of length K lists, where M is the amount of classifiers,
     N is the amount of data instances, and K is tha amount of classes
    :param expected: the actual labels of the dataset which the clfs were trying to predict.
    :type expected: array-like
    :param ENSEMBLE_SIZES: for each integer k in ENSEMBLE scores, all ensembles of length k are tried, and the best ones are reported.
    :type ENSEMBLE_SIZES: list of int
    :return: ensemble_scores, modified, where ensemble_scores[ensemble] = ensemble_scores[ensemble] + acc_for_this_dataset
    (Note that it's unnormalized, and should be normalized in another function)
    :rtype: dict
    """
    N_classifiers = len(clf_prediction_probs)
    ensembles = get_ensembles(ENSEMBLE_SIZES, N_classifiers)
 
    for ensemble in ensembles:
        ensemble = tuple(ensemble)
        acc = get_ensemble_accuracy(ensemble, clf_prediction_probs, expected)
        if ensemble not in ensemble_scores:
            ensemble_scores[ensemble] = 0
        ensemble_scores[ensemble] +=acc

    return ensemble_scores



def display_ensemble_accuracies(ensemble_scores, clf_names, N_datasets, ensemble_training_scores):
    """
    """
    # normalize ensemble_scores and ensemble_training_accs and split dictionary into lists of values and keys (for sorting):
    return_string = ""
    try:
        ensemble_accs = []
        ensemble_ids = ensemble_scores.keys()
        for key in ensemble_ids:
            ensemble_accs.append(ensemble_scores[key]*1.0 / N_datasets)

        ensemble_training_accs = []
        ensemble_training_scores_ids = ensemble_training_scores.keys()
        for key in ensemble_training_scores_ids:
            ensemble_training_accs.append(ensemble_training_scores[key]*1.0 / N_datasets)

    
        std_ensemble_scores = np.argsort(ensemble_accs)

    
        return_string += "Top ensemble classifiers:\n"
        for i in range(ENSEMBLES_TO_DISPLAY):
            idx = std_ensemble_scores[-i-1]
            return_string += "%s: %.1f"%(" + ".join(np.array(clf_names)[np.array(ensemble_ids[idx])]), ensemble_accs[idx]*100)
            return_string += u"    (TA:%.1f%%)\n"%(ensemble_training_accs[idx]*100)
    except Exception as e:
        return_string += str(e)
        return_string += "\n"
        return_string += str(traceback.format_exc())

    return return_string


def write_results_to_doc_file(doc_file, results, dataset_language, clf_names, ensemble_scores, N_datasets, total_data_instances, t_start, training_errors, ensemble_training_accs, args):
    """
    good for running tests over the weekend (using screen) and coming back to a nicely written file full of results!
    """
    df = open(doc_file, "a")
    df.write("="*100)
    df.write("\n")
    df.write("+"*100)
    #df.write("\nUSING ANALYZER!!!!!!\n")
    df.write("\nRun completed on %s, with following parameters:\nSAMPLE_SIZE: %s\nFEATURE_SELECT: %s\n"\
                 %(time.asctime(), SAMPLE_SIZE, FEATURE_SELECT))
    df.write("CSV_FOLDER: %s\nFEATURES_FOLDER: %s\nuse_numeric_features: %s\n"\
                 %(args.csv_folder, args.features_folder, args.use_numeric_features))
    df.write("Languages: %s\n"%LANGUAGES_TO_ITERATE_OVER)
    df.write(display_stats(results, dataset_language, clf_names, training_errors).encode('utf-8'))
    df.write(display_ensemble_accuracies(ensemble_scores, clf_names, N_datasets, ensemble_training_accs).encode('utf-8'))
    df.write("Total amount of data instances %s\n"%total_data_instances)
    df.write("Program took %s seconds to execute.\n\n"%(time.time() - t_start))
    df.close()

def get_ensembles(ensemble_sizes, N_classifiers):
    # ensembles could be calculated outside of the clients of this function and passed in, but I thought this was easier.  It is 
    # an iterable of iterables, each of which is a subse of classifiers, by index.
    ensembles = it.chain(*[it.combinations(range(N_classifiers), ensemble_size) for ensemble_size in ensemble_sizes])
    return ensembles
    #    unflattened_ensembles = ((tuple(combo) for combo in itertools.combinations(predictions, ensemble_size)) for ensemble_size in ENSEMBLE_SIZES)
    #   ensembles = (ensemble for ensemble_class in unflattened_corpus for ensemble in ensemble_class)


def main():
    print "Args: %s"%sys.argv
    args = get_args()
    t_start = time.time()
    dataset_language = [] # languages of each of the datasets in classfile_folder
    results = [] # classifier accuracies for each classfile in classfile_folder
    training_errors = [] # For diagnosing whether the model suffers from over- or underfitting
    DEBUG_SKIP = 0
    total_data_instances = 0
    N_datasets = 0
    N_files_read = 0
    ENSEMBLE_SIZES = [2, 3, 4, 5] # eventually make this a global constant
    ensemble_scores = {} # keys are tuples of strings representing classifiers.  Values are sums of accuracies.
    # (thus will need to be divided by the amount of datasets after this coming for loop)
    # ensemble_scores = [0]*[sum(sc.comb(M, k) for k in ENSEMBLE_SIZES] # where M is the amount of classifiers
    ensemble_training_accs = {} # These will be converted into measures of error rather than accuracy, but it's easier to calculate so.
    global FEATURE_SELECT # TODO this globality is temporary!!!
    FEATURE_SELECT = args.feature_select
    global SAMPLE_SIZE
    SAMPLE_SIZE = args.sample_size

    for filename in os.listdir(args.csv_folder): #might be cleaner to iterate through languages

        N_files_read +=1
        if filename == ".DS_Store": # macs are odd
            print "Encountered .DS_Store file...skipping"
            continue
        
        language_name = filename.split('-')[1].split('.')[0].strip()  # This gets the name from the CSV file
        language_code = LANGUAGE_NAME_TO_CODE[language_name]
        if language_name not in LANGUAGE_NAME_TO_CODE.keys():
            print "WARNING BAD BAD!: Encountered classfile for language %s, for which we have not extracted features.  Skipping."%language_name
        if language_name not in LANGUAGES_TO_ITERATE_OVER:
            continue
        if N_files_read <=DEBUG_SKIP:
            continue
        else:
            N_datasets +=1
        
        processed_flname = "%s/%s_feats/processed.csv"%(args.features_folder, language_code) # for numeric features
        unprocessed_flname = "%s/%s_feats/unprocessed.csv"%(args.features_folder, language_code) # for linguistic features
            
        # note that, since we call the transform function directly, there is no need for
        # an outfile.  This saves time and space!
        # Note: below actually a list, not a file.
        class_file = transform(open('%s/%s'%(args.csv_folder, filename), 'r'), for_secondary = False).split('\n')
        features_file = open(unprocessed_flname, 'r')
        numeric_features_file = open(processed_flname, 'r')

        print "Testing classifier accuracy for %s: "%language_name
        # test.get_classifier_accuracies returns a list of tuples of the form (name_of_classifier, pct_accuracy, runtime)
        #TODO: Make Thread (would require rewriting 'append' statements)
        accuracies, training_error = \
            test.get_classifier_accuracies(class_file=class_file, features_file = features_file, numeric_features_file = numeric_features_file,\
                                               verbose = args.verbose, feature_select = FEATURE_SELECT, sample_size = SAMPLE_SIZE,\
                                               return_ensemble_materials=True, return_training_error = True, use_numeric_features = args.use_numeric_features)

        accuracies = [tup for tup in accuracies if tup !=None] # In case a classifier throws an error, like QDA was doing.
        clf_prediction_probs = [tup[3] for tup in accuracies]
        clf_actual_labels = accuracies[0][4]
        results.append(accuracies)

        training_error = [tup for tup in training_error if tup !=None] 
        training_prediction_probs = [tup[2] for tup in training_error]
        training_labels = training_error[0][3]
        training_labels = [[label] for label in training_labels]
        
        training_errors.append(training_error)
        # NOTE: the only difference between training_labels and clf_actual_labels is that the latter may 
        # have excluded certain class labels because of the LOOCV

        # dataset_language.append(language_name_to_code[language_name])
        dataset_language.append(language_name)
        if SAMPLE_SIZE:
            total_data_instances += SAMPLE_SIZE
        else: #TODO/Note: this is actually wrong (an over-estimate), because the 'file' contains empty entries
            #which are discarded by test.get_classifier_accuracies.  Namely, 24 of them.
            total_data_instances += len(class_file)-1
        
        #ENSEMBLE CLASSIFICATION
        # This function will udate ensemble_scores.  At the end, the best ensembles will be reported. 
        # accuracies[0] might as well have been accuracies[1].  It's an arbitrary tuple, because 
        # the expectations (actual labels) are the same across a data instance.

        ensemble_scores = predict_all_ensembles_precalc(ensemble_scores, clf_prediction_probs, clf_actual_labels, ENSEMBLE_SIZES)
        ensemble_training_accs = predict_all_ensembles_precalc(ensemble_training_accs, training_prediction_probs, training_labels, ENSEMBLE_SIZES)

    clf_names = [tup[0] for tup in results[0]]
    
    print display_stats(results, dataset_language, clf_names, training_errors)
    print display_ensemble_accuracies(ensemble_scores, clf_names, N_datasets, ensemble_training_accs)

    print "Total amount of data instances %s\n"%total_data_instances
    print "Program took %s seconds to execute."%(time.time() - t_start)
    
    if args.doc_file:
        write_results_to_doc_file(args.doc_file, results, dataset_language, clf_names, ensemble_scores, N_datasets, total_data_instances, t_start, training_errors, ensemble_training_accs, args)





if __name__ == u'__main__':
    main()
