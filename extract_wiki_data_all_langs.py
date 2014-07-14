"""
File: extract_wiki_data_all_langs
@author: Isaac Caswell
@Created: 11 July 2014

Runs the extract_wiki_data script on a variety of languages, and saves them as files in a directory as dirname/[language code].csv. 

"""

import extract_wiki_data as ewd
import sys
import os


LANGUAGES = ["de","en","zh","fr","ja","pl","pt","ru","es"]


def main():
    dir_name = "wiki_data_all" 
    if len(sys.argv) == 2:
        dir_name = sys.argv[1]

    if not os.path.exists(dir_name): #note: this can fail in race conditions, but that's liable never to be a problem here....
        os.makedirs(dir_name)
    
    for lang in LANGUAGES:
        print "extracting features for %s...."%lang
        ewd.extract_features_and_write_to_file(outfile = u'%s/%s.csv'%(dir_name, lang), lang = lang)
    pass

if __name__ == u'__main__':
    main()
