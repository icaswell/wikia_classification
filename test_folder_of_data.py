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
  [python extract_wiki_data_all_langs.py]
  python test_folder_of_data.py [CSV_folder_name]

Arguments:
  -classfile_folder: name of a folder (in current directory) of CSV files (classfiles)

Known bugs/problems:
  -test_classifiers barfs up a lot of UserErrors and RuntimeErrors, namely that log encountered an 
  invalid value in divide
  -unfortunate name
"""
import time
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


#GET_DATASET == 1  #if the features are to be extracted by the script (extract_wiki_ids_all_langs) #can't do this because it's multithreaded

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

LANGUAGES_TO_ITERATE_OVER = ["German", "English", "Portuguese", "Spanish"] #,"Russian", "Polish", "Japanese", "French", "Chinese"]


def get_args():
    ap = ArgumentParser()
    ap.add_argument('--class-folder', default = "wikia_classes", dest='class_folder')
    ap.add_argument('--features-folder', default = "wiki_data_all", dest='features_folder')
    ap.add_argument('--verbose', dest = 'verbose', action = 'store_true', default = False)
    return ap.parse_args()


def display_stats(results, dataset_language, clf_names):
    #best cassifier, on average:
    # below line makesnames same number of characters, for prettiness
    accuracies = [[tup[1] for tup in dataset] for dataset in results]
    avg_acc = [np.mean(np.array(i)) for i in zip(*accuracies)]
    sd_acc = [np.std(np.array(i)) for i in zip(*accuracies)]

    sorted_avg_acc_and_name = sorted(zip(avg_acc, clf_names, sd_acc), reverse=True)
    print "Average performance: "
    for tup in sorted_avg_acc_and_name:
        print u"\t%s %s \u00B1 %.3f"%(tup[1] + (17 - len(tup[1]))*" ", tup[0], tup[2]) # where 17 is the length of the longest name, plus 1, for prettiness of formatting
   
        
    #language: [best classifiers]
    print "\nLineup per language: "
    lineup_per_language = [sorted(zip(acc, clf_names), reverse=True) for acc in accuracies] 
    for i in range(len(dataset_language)):
        print "%s: "%dataset_language[i],
        for tup in lineup_per_language[i]:
            print "%s (%s),"%(tup[1], tup[0]),
        print "\n",
    


def get_ensemble_accuracy(ensemble, all_prediction_probs, expected):
    """
    :param ensemble: a tuple of classifier indices
    :type ensemble: tuple
    :param all_prediction_probs:  all_prediction_probs[i][j][0][k] is the probability, using classifier i, that data instance j is in class k
    :type all_prediction_probs: list
    :param expected: length N list of actual labels of the data instance
    :type expected: list
    :return: percent accuracy of given ensemble over the data instance
    :rtype: float
    """
    N_classifiers = len(all_prediction_probs)
    N_instances = len(all_prediction_probs[0])
    N_classes = len(all_prediction_probs[0][0][0])
    ensemble_pred_probs_uncollapsed = np.squeeze(np.array(all_prediction_probs)[np.array(ensemble)])
    #If this were in numpy to start with, this would be much easier...
    
    ensemble_pred_probs = np.zeros((N_instances, N_classes))
    for clf_preds in ensemble_pred_probs_uncollapsed:
        ensemble_pred_probs[:, :] += np.array(clf_preds)
    
    predictions = list(ensemble_pred_probs.argmax(axis=1))
    return sum([p==e[0] for p, e in zip(predictions, expected)])*1.0/len(expected)


def predict_all_ensembles_precalc(ensemble_scores, clf_prediction_probs, expected, ENSEMBLE_SIZES):
    # TODO: implement complete bootstrap bagging ensemble classification. (this one uses all training instance on every run.)
    # This however would necessitate going deeper into the code, probably messing with the function from __init__ rather than 
    # the "precalc" variation

    # :param: ensemble_scores: keys are tuples of strings representing classifiers.  Values are sums of accuracies.
    # :type: dict

    # predictions is a length M list of length N lists, where M is the amount of classifiers, and N is the amount of data instances
    # predictions[i][j] is the predicted label for instance j from classifier i.
    
    # determine ensemble predictions of all subsets by summing predictions (yes, they needed)
    # calculate accuracy by comparing to expectations
    # return to be printed.  aggregate in another function.
    # "precalc" refers to the fact that the predictions of each classifier have already been calculated.
    
    N_classifiers = len(clf_prediction_probs)
    ensembles = it.chain(*[it.combinations(range(N_classifiers), ensemble_size) for ensemble_size in ENSEMBLE_SIZES])
    #    unflattened_ensembles = ((tuple(combo) for combo in itertools.combinations(predictions, ensemble_size)) for ensemble_size in ENSEMBLE_SIZES)
    #   ensembles = (ensemble for ensemble_class in unflattened_corpus for ensemble in ensemble_class)

    for ensemble in ensembles:
        ensemble = tuple(ensemble)
        acc = get_ensemble_accuracy(ensemble, clf_prediction_probs, expected)
        if ensemble not in ensemble_scores:
            ensemble_scores[ensemble] = 0
        ensemble_scores[ensemble] +=acc

    return ensemble_scores

def display_ensemble_accuracies(ensemble_scores, clf_names, N_datasets):
    # normalize ensemble_scores and split dictionary into lists of values and keys (for sorting):
    ensemble_accs = []
    ensemble_ids = ensemble_scores.keys()
    for key in ensemble_ids:
        ensemble_accs.append(ensemble_scores[key]*1.0 / N_datasets)
    
    std_ensemble_scores = np.argsort(ensemble_accs)

    print "Top ensemble classifiers:"
    for i in range(10):
        idx = std_ensemble_scores[-i-1]
        print "%s: %.3f"%("+ ".join(np.array(clf_names)[np.array(ensemble_ids[idx])]), ensemble_accs[idx])



def main():
    # if GET_DATASET:
    #    ewdal.extract_wiki_data_all_langs(dirname = FEATURE_DATA_DIR_NAME, langs = LANGUAGE_NAME_TO_CODE.values())
    args = get_args()
    t_start = time.time()
    dataset_language = [] # languages of each of the datasets in classfile_folder
    results = [] # classifier accuracies for each classfile in classfile_folder
    DEBUG_SKIP = 0
    total_data_instances = 0
    N_datasets = 0
    N_files_read = 0
    ENSEMBLE_SIZES = [1, 2, 3, 4, 9] # eventually make this a global constant
    ensemble_scores = {} # keys are tuples of strings representing classifiers.  Values are sums of accuracies.
    # (thus will need to be divided by the amount of datasets after this coming for loop)
    # ensemble_scores = [0]*[sum(sc.comb(M, k) for k in ENSEMBLE_SIZES] # where M is the amount of classifiers
    for filename in os.listdir(args.class_folder): #might be cleaner to iterate through languages
        N_files_read +=1
        if filename == ".DS_Store": # macs are odd
            print "Encountered .DS_Store file...skipping"
            continue
        
        language_name = filename.split('-')[1].split('.')[0].strip()
        if language_name not in LANGUAGE_NAME_TO_CODE.keys():
            print "WARNING BAD BAD!: Encountered classfile for language %s, for which we have not extracted features.  Skipping."%language_name

        if language_name not in LANGUAGES_TO_ITERATE_OVER:
            continue

        if N_files_read <=DEBUG_SKIP:
            continue
        else:
            N_datasets +=1



        # note that, since we call the transform function directly, there is no need for
        # an outfile.  This saves time and space!
        class_file = transform(open('%s/%s'%(args.class_folder, filename), 'r'), for_secondary = False).split('\n')
        features_file = open("%s/%s.csv"%(args.features_folder, LANGUAGE_NAME_TO_CODE[language_name]), 'r') # Highly uncertain as to whether this is right or not

        print "Testing classifier accuracy for %s: "%language_name
        # test.get_classifier_accuracies returns a list of tuples of the form (name_of_classifier, pct_accuracy, runtime)
        accuracies = test.get_classifier_accuracies(class_file=class_file, features_file = features_file, verbose = False) #TODO: Make Thread (would require rewriting 'append' statements)
        accuracies = [tup for tup in accuracies if tup !=None] # I don't where the None entry came from, but it was messing things up
        results.append(accuracies)
        # dataset_language.append(language_name_to_code[language_name])
        dataset_language.append(language_name)
        total_data_instances += len(class_file)
        
        #ENSEMBLE CLASSIFICATION
        # This function will udate ensemble_scores.  At the end, the best ensembles will be reported. 
        # accuracies[0] might as well have been accuracies[1].  It's an arbitrary tuple, because 
        # the expectations (actual labels) are the same across a data instance.
        clf_prediction_probs = [tup[3] for tup in accuracies]
        ensemble_scores = predict_all_ensembles_precalc(ensemble_scores, clf_prediction_probs, accuracies[0][4], ENSEMBLE_SIZES)
        

    clf_names = [tup[0] for tup in results[0]]
    
    display_stats(results, dataset_language, clf_names)
    display_ensemble_accuracies(ensemble_scores, clf_names, N_datasets)

    print "Total amount of data instances %s"%total_data_instances
    print "Program took %s seconds to execute."%(time.time() - t_start)




def old_main():
    t_start = time.time()
    CSV_folder = sys.argv[1]
    dataset_language = [] # languages of each of the datasets in classfile_folder
    results = [] # classifier accuracies for each classfile in classfile_folder
    DEBUG_SKIP = 0
    total_data_instances = 0
    N_datasets = 0
    N_files_read = 0
    ENSEMBLE_SIZES = [1, 2, 3, 4, 9] # eventually make this a global constant
    ensemble_scores = {} # keys are tuples of strings representing classifiers.  Values are sums of accuracies.
    # (thus will need to be divided by the amount of datasets after this coming for loop)
    # ensemble_scores = [0]*[sum(sc.comb(M, k) for k in ENSEMBLE_SIZES] # where M is the amount of classifiers
    for filename in os.listdir(CSV_folder):
        N_files_read +=1
        if filename == ".DS_Store": # macs are odd
            print "Encountered .DS_Store file...skipping"
            continue

        
        if N_files_read <=DEBUG_SKIP:
            continue
        else:
            N_datasets +=1

        # note that, since we call the transform function directly, there is no need for
        # an outfile.  This saves time and space!
        class_file = transform(open('%s/%s'%(CSV_folder, filename), 'r'), for_secondary = False).split('\n')
        features_file = transform(open('%s/%s'%(CSV_folder, filename), 'r'), for_secondary = True).split('\n') # TODO! this is wrong.
        language_name = filename.split('-')[1].split('.')[0].strip()

        print "Testing classifier accuracy for %s: "%language_name
        # test.get_classifier_accuracies returns a list of tuples of the form (name_of_classifier, pct_accuracy, runtime)
        accuracies = test.get_classifier_accuracies(class_file=class_file, features_file = features_file, verbose = False)
        accuracies = [tup for tup in accuracies if tup !=None] # I don't where the None entry came from, but it was messing things up
        results.append(accuracies)
        # dataset_language.append(language_name_to_code[language_name])
        dataset_language.append(language_name)
        total_data_instances += len(class_file)
        
        #ENSEMBLE CLASSIFICATION
        # This function will udate ensemble_scores.  At the end, the best ensembles will be reported. 
        # accuracies[0] might as well have been accuracies[1].  It's an arbitrary tuple, because 
        # the expectations (actual labels) are the same across a data instance.
        clf_prediction_probs = [tup[3] for tup in accuracies]
        ensemble_scores = predict_all_ensembles_precalc(ensemble_scores, clf_prediction_probs, accuracies[0][4], ENSEMBLE_SIZES)
        

    clf_names = [tup[0] for tup in results[0]]
    
    display_stats(results, dataset_language, clf_names)
    display_ensemble_accuracies(ensemble_scores, clf_names, N_datasets)

    print "Total amount of data instances %s"%total_data_instances
    print "Program took %s seconds to execute."%(time.time() - t_start)

if __name__ == u'__main__':
    main()
