"""
Usage: 
  python extract_wiki_data.py --outfolder en.csv --lang en

Where outfolder is a folder containing files called unprocessed.csv and processed.csv.
The first will be tokens, i.e. a bag of words.  The second will be a list of numeric data.
"""
import os, sys, inspect # this all needed for Chinese segmentation
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]) + "/smallseg_0.6")

if cmd_subfolder not in sys.path:
     sys.path.insert(0, cmd_subfolder)

from argparse import ArgumentParser
from collections import OrderedDict
from text.blob import TextBlob
from nltk.util import bigrams
from multiprocessing import Pool
from traceback import format_exc
import nltk.stem.snowball as snowball # Mod. Isaac [for foreign language parsing?]

from nltk.tokenize.regexp import WhitespaceTokenizer
from smallseg import SEG
from re import compile as _Re

from nltk.corpus import stopwords
from boto import connect_s3
import requests
import codecs
import traceback
import wikipedia #TODO: before full automatization, use better API.
import re
import string

# stops, tokenizer and stemmer (still global) were originally declared out here.  I moved them into the main method
# so they could be taylored to the language.
code_to_language_name = {'en': u'english',
                    'fr': u'french',
                    'zh': u'chinese',
                    'de': u'german',
                    'ru': u'russian',
                    'pl': u'polish',
                    'ja': u'japanese',
                    'pt': u'portuguese',
                    'es': u'spanish'
                    }

MIN_WORD_LENGTH = 4 # Program changes to 1 for Chinese and Japanese


def get_wikipedia_title(wikia_title):
     search_results = wikipedia.search(wikia_title)
     if search_results == []:
          return "NO_SEARCH_RESULTS_FOUND"
     else: 
          return search_results[0]


def get_wikipedia_desc_from_wikia_title(wikia_title):
     wikia_title = re.sub(" Wiki", "", wikia_title)
     try:
          wikia_title =  get_wikipedia_title(wikia_title)
          wikipedia_page = wikipedia.page(get_wikipedia_title(wikia_title))
          
          #return wikipedia_page.summary.encode('utf-8').split('\n')[0]
          return wikipedia_page.summary.encode('utf-8')
     except Exception as e:
          print "\nERROR with %s:"%wikia_title
          print str(e).split('\n')[0]
          print wikia_title
          #Even the disambiguation desc. (which we assume the error to be) may
          #have something like Violeta (TV) in it!!!  We want that 'TV'!
          disambig_string = str(e)#.decode('utf-8')
          #gzorp
          disambig_string = re.sub(wikia_title, "", disambig_string)
          return disambig_string

     #return wikipedia_page.summary



def get_args():
    ap = ArgumentParser()
    ap.add_argument(u'--num-processes', dest=u"num_processes", default=8, type=int)
    ap.add_argument(u'--solr-host', dest=u"solr_host", default=u"http://search-s10:8983") # TODO: what does 8983 mean?
    ap.add_argument(u'--outfolder', dest=u'outfolder', default=u'wiki_features_data') #modified by Isaac.  Alas I cannot account for
    #the language in this version.  Fortunately, no one needs to call this from command line any more.
    ap.add_argument(u'--s3dest', dest=u's3dest')
    ap.add_argument(u'--lang', dest=u'lang', default = u'en')
    return ap.parse_args()

"""
def get_wiki_data(args):
   # Gets wiki data as JSON docs for all English wikis with 50 or more articles (content pages).

    fl = u'id,top_categories_mv_%s,hub_s,top_articles_mv_%s,description_txt,sitename_txt'%(LANGUAGE, LANGUAGE)
    params = {u'fl': fl,
              u'start': 0,
              u'wt': u'json',
              u'q': u'lang_s:%s AND articles_i:[50 TO *]'%LANGUAGE, #lang_s is not working properly
              u'rows': 500}
    data = []
    while True:
        response = requests.get(u'%s/solr/xwiki/select' % args.solr_host, params=params).json()
        data += response[u'response'][u'docs']
        if response[u'response'][u'numFound'] < params[u'rows'] + params[u'start']:
            return OrderedDict([(d[u'id'], d) for d in data])
        params[u'start'] += params[u'rows']

"""
def get_mainpage_text(args, wikis):
    """
    Get mainpage text for each wiki
    
    now modified to get other numerical features, as well as wikipedia features.
    
    seems to be confusing and redundant to have this as a separate function than get_wikidata_from _list!

    wikis, an ordered dict created by get_wiki_data, is modified so each value (itself a dict)
    also contains the key-value pair ('main_page_text', [mainpage text]) 
    :param wikis: our wiki data set
    :type wikis:class:`collections.OrderedDict`
    :return: OrderedDict of search docs, id to doc, with mainpage raw text added
    :rtype:class:`collections.OrderedDict`
    """
    for i in range(0, len(wikis), 100):
        query = u'(%s) AND is_main_page:true' % u' OR ' .join([u"wid:%s" % wid for wid in wikis.keys()[i:i+100]])
        fl = u'wid,html_%s,wikiviews_monthly,wam'%LANGUAGE # commented out 29 July 2014
        #fl = u'wid,html_%s,wikiviews_monthly,wam,edits_i,videos_i,promoted_b,active_users_i,alldomains_mv_wd'%LANGUAGE #WHY NO WORK
        params = {u'wt': u'json',
                  u'start': 0,
                  u'rows': 100,
                  u'q': query,
                  u'fl': fl}
        response = requests.get(u'%s/solr/main/select' % args.solr_host, params=params).json()
        for result in response[u'response'][u'docs']:
            if u'html_%s'%LANGUAGE in result: #added by Isaac.  For some reason, French and Chinese had some entries without this key.
                wikis[str(result[u'wid'])][u'main_page_text'] = result[u'html_%s'%LANGUAGE]
                wikis[str(result[u'wid'])][u'wam'] = result[u'wam']
                wikis[str(result[u'wid'])][u'wikiviews_monthly'] = result[u'wikiviews_monthly']
    return wikis


def normalize(wordstring):
    global stemmer, tokenizer, stops
    try:
        return [stemmer.stem(word.strip(string.punctuation)) for word in tokenizer.tokenize(wordstring.lower())
                if len(word) >= MIN_WORD_LENGTH and word not in stops] # Added by Isaac to handle i.e. Japanese
    except (Exception, IndexError) as e:
        traceback.format_exc()
        print e
        print wordstring
        print tokenizer.tokenize(wordstring.lower())


def wiki_to_feature(wiki):
    """
    Specifically handles a single wiki document
    
    NOTE!!!!!  I,saac changed TOP_ART:, TOP_CAT:, DESC: and ORIGINAL_HUB: to 
    TOP_ART_, TOP_CAT_, DESC_ and ORIGINAL_HUB_ so they don't get split up in the tokenizer later on.

    As it was before, these features were split by a tokenizer later on.  

    
    Also added Wikipedia features
    
    :param wiki: dict for wiki fields
    :type wiki: dict
    :return: tuple with wiki id and list of feature strings
    :rtype: tuple
    """
    #try:
    features = [] # strings representing categories, art, old hubs, etc
    numeric_features = [] # wam, pg count, hub ct, etc ADDED BY ISAAC
    bow = []
    features += [u'ORIGINAL_HUB_%s' % wiki.get(u'hub_s', u'')]
    
    features += [u'TOP_CAT_%s' % u'_'.join(normalize(c)) for c in wiki.get(u'categories_mv_%s'%LANGUAGE, [])] # Added lang --Isaac
    bow +=                      [u"_".join(normalize(c)) for c in wiki.get(u'categories_mv_%s'%LANGUAGE, [])] # added lang. --Isaac
    features += [u'TOP_ART_%s' % u"_".join(normalize(a)) for a in wiki.get(u'top_articles_mv_%s'%LANGUAGE, [])] # added lang. --Isaac
    bow +=                      [u"_".join(normalize(a)) for a in wiki.get(u'top_articles_mv_%s'%LANGUAGE, [])]

    


    desc_ngrams = [u"_".join(n) for grouping in
                   [bigrams(normalize(np))
                    for np in TextBlob(wiki.get(u'description_txt', [u''])[0]).noun_phrases]
                   for n in grouping]

    # Note: splitting the descriptions is likely more meaningful, because if they are unplit they contain tokens such as book are book_seri
    # Problem tho: comic_book and book_seri would map to same thing :(, as would book_seri and seri_star.  Thus full descs must also be added, 
    split_descs = [d for ngram in desc_ngrams for d in ngram.split("_")]  

    bow += desc_ngrams
    features += [u'DESC_%s' % d for d in desc_ngrams]
    #features += [u'SPLIT_DESC_%s' % d for d in split_descs]
    bow += split_descs
    
    bow += [u"_".join(b) for b in bigrams(normalize(wiki[u'sitename_txt'][0]))]
    
    #mp_nps = TextBlob(wiki.get(u'main_page_text', u'')).noun_phrases
    mp_nps = normalize(wiki.get(u'main_page_text', u''))
    print mp_nps
    bow += [u"_".join(bg) for grouping in [bigrams(normalize(n)) for n in mp_nps] for bg in grouping]
    bow += [u''.join(normalize(w)) for words in [np.split(u" ") for np in mp_nps] for w in words]
    
    #WIKIPEDIA FIRST PARAGRAPHS!
    wikipedia_desc = get_wikipedia_desc_from_wikia_title(wiki[u'sitename_txt'][0])    
    wikipedia_desc = wikipedia_desc.decode("utf-8")
    
    #mp_nps = TextBlob(wikipedia_desc).noun_phrases
    mp_nps = normalize(wikipedia_desc)
    bow += [u"_".join(bg) for grouping in [bigrams(normalize(n)) for n in mp_nps] for bg in grouping]
    bow += [u''.join(normalize(w)) for words in [np.split(u" ") for np in mp_nps] for w in words]

    #EXTRALINGUISTIC NUMERIC FEATURES

    numeric_features.append(wiki.get(u'wam'))
    numeric_features.append(wiki.get(u'wikiviews_monthly'))


    return wiki[u'id'], (bow + features, numeric_features)
    # return wiki[u'id'], bow + features # OLD
    #except Exception as e:
    #    print e, format_exc()
    #    raise e


def wikis_to_features(args, wikis):
    """
    Turns wikis into a set of features
    :param args:argparse namespace
    :type args:class:`argparse.Namespace`
    :param wikis: our wiki data set
    :type wikis:class:`collections.OrderedDict`
    :return: OrderedDict of features, id to featureset
    :rtype:class:`collections.OrderedDict`
    """
    p = Pool(processes=args.num_processes)
    return OrderedDict(p.map_async(wiki_to_feature, wikis.values()).get())


def get_wiki_data_from_list(args, wid_list):
    """
    A copy of get_wiki_data, except that this one gets data only from those articles specified by the id_list, 
    instead of the full cerca 20k gotten by get_wiki_data
    """
    its = 0
    step = 99
    data = []
    while True: # not getting the request all in one go because that can give a 413 error (request too large)
         if its>=len(wid_list):
              break
         query = u' OR ' .join([u"id:%s" % wid for wid in wid_list[its:its+step]])
         #fl = u'id,top_categories_mv_%s,hub_s,top_articles_mv_%s,description_txt,sitename_txt,wikiviews_monthly, wam'%(LANGUAGE, LANGUAGE)
         #fl = u'id,top_categories_mv_%s,hub_s,top_articles_mv_%s,description_txt,sitename_txt'%(LANGUAGE, LANGUAGE) #commented out 29 July 2014
         fl = u'id,categories_mv_%s,hub_s,top_articles_mv_%s,description_txt,headline_txt,sitename_txt'%(LANGUAGE, LANGUAGE)
         its +=step
         params = {u'fl': fl,
                   u'start': 0,
                   u'wt': u'json',
                   u'q': query,
                   u'rows': 500}
         while True:
              #response = requests.get(u'%s/solr/xwiki/select' % args.solr_host, params=params)
              response = requests.get(u'%s/solr/xwiki/select' % args.solr_host, params=params).json()
              data += response[u'response'][u'docs']

              if response[u'response'][u'numFound'] < params[u'rows'] + params[u'start']:
                   break
              params[u'start'] += params[u'rows']

    return OrderedDict([(d[u'id'], d) for d in data])



def init_args(num_processes, solr_host, outfolder, s3dest):
    """
    this rigamarole is so that this module can be called from command line and from scripts.  calling from command line, 
    argparse makes an args object.  This function simulates that.
    """
    class ArgumentObject: # TODO: the default for outfolder cannot account for lang, so cannot be used.  default should be nixed.
        def __init__(self, num_processes=8, solr_host = u"http://search-s10:8983", outfolder = u'wiki_features_data', s3dest = None): #lang = u'en'):
            self.num_processes = num_processes
            self.solr_host = solr_host
            self.outfolder = outfolder
            self.s3dest = s3dest
    return ArgumentObject(num_processes, solr_host, outfolder, s3dest)

def get_tokenizer(lang):
    class ChineseTokenizer(): # a wrapper class to support the duck typing
        def __init__(self):
            self._seg = SEG()
        def tokenize(self, text):
            wlist = self._seg.cut(text)
            #wlist.reverse() Using Bag of W assumption, so unneeded, yeah?
            return wlist

    class UnicodeTokenizer():
        def __init__(self):
            self._splitter = _Re( '(?s)((?:[\ud800-\udbff][\udc00-\udfff])|.)' ).split
        def tokenize(self, text):
            return [ch for ch in self._splitter(text) if ch]

    if lang == "zh":
        #return UnicodeTokenizer()
        return ChineseTokenizer()
    if lang == "ja":
         #return ChineseTokenizer()
         return UnicodeTokenizer()
    else:
        return WhitespaceTokenizer()

        
def get_stemmer(lang):
    """
    returns a stemmer for the appropriate language, or a dummy object.
    The dummy object simply returns an unstemmed word when the .stem(word) 
    method is called. This model actuall makes sense with an isolating language
    like Chinese, but alas will do Polish and Japanese a disservice.
    """
    class NonStemmer():
        def __init__(self):
            pass
        def stem(self, word):
            return word

    return { #Yay python pseudo switch!
        'en': snowball.EnglishStemmer(),
        'es': snowball.SpanishStemmer(),
        'fr': snowball.FrenchStemmer(),
        'de': snowball.GermanStemmer(),
        'pt': snowball.PortugueseStemmer(),
        'ru': snowball.RussianStemmer()
        }.get(lang, NonStemmer())

def extract_features_from_list_of_wids(wid_list, num_processes=8, solr_host = u"http://search-s10:8983", outfolder = None, lang = u'en', s3dest = None):
    """
    for each id in wid_list, extracts content from the corresponding wiki, parses it into meaningful 
    features and saves it to the specified file.  A replacement for extract_features_and_write_to_file 
    (which contains the functionality from Robert's original script) which was grossly inefficient,
    scraping around 20,000 wikis, when only 100-200 are needed per language (100x space storage), and 
    ommitting smaller wikis, which are useful so sample for randomization porpoises.
    """

    #TODO: label outfolder such that it's clear that it's a microfolder of two files
    if not outfolder:
        outfolder = u"wiki_features_data_%s"%lang
    
    args = init_args(num_processes, solr_host, outfolder, s3dest)

    global LANGUAGE
    LANGUAGE = lang # I use a global because otherwise it's a pain to make map_async accept multiple arguments :( --Isaac
    global stops
    stops = stopwords.words(code_to_language_name[LANGUAGE]) # Added --Isaac.  kept global, as Robert had written it, but moved it here so the language could be specified.
    stops = [stop.decode('utf-8') for stop in stops] # tohandle i.e. Japanese. added by Isaac

    global stemmer
    stemmer = get_stemmer(lang)
    
    global tokenizer
    tokenizer = get_tokenizer(lang)
    
    if lang in [u'zh', u'ja']:
        global MIN_WORD_LENGTH
        MIN_WORD_LENGTH = 1
        print u"set minium word length to 1, because we're dealing with %s here..."%code_to_language_name[lang]
    # features: mapping of wid to (linguistic feats, numeric feats)
    features = wikis_to_features(args, get_mainpage_text(args, get_wiki_data_from_list(args, wid_list))) 
    #flname = args.s3dest if args.s3dest else args.outfile # Robert wrote this line and I don't have any idea what its porpoise is
    # NUMERIC FEATURES_TODO: delete above line, make directory if needed 


    if not os.path.exists(args.outfolder): #note: this can fail in race conditions, but that's liable never to be a problem here....
        os.makedirs(args.outfolder)
    processed_flname = "%s/processed.csv"%args.outfolder # for numeric features
    unprocessed_flname = "%s/unprocessed.csv"%args.outfolder # for linguistic features
    with codecs.open(unprocessed_flname, u'w', encoding=u'utf8') as unprocessed_fl, \
             codecs.open(processed_flname, u'w', encoding=u'utf8') as processed_fl:
        for wid, features in features.items():
            line_for_writing_unprocessed = u",".join([wid, u",".join(features[0])]) + u"\n"
            unprocessed_fl.write(line_for_writing_unprocessed)

            line_for_writing_processed = u",".join([wid, u",".join([str(i) for i in features[1]])]) + u"\n"
            processed_fl.write(line_for_writing_processed)
 
    if args.s3dest: # No idea
        b = connect_s3().get_bucket(u'nlp-data')
        k = b.get_key(args.s3dest)
        k.set_contents_from_filename(args.s3dest)

    #1. GWD equivalent
    #2. get_mainpage_text with'm


    

def main():
     print "U better be running this as a test, because this just has some hardcoded list it's calling.  Functionality\
 of main() has been exported to the callable methods extract_features_from_list_of_wids(), or if you're crazy, extract_features_and_write_to_file()"
     
     args = get_args()
     extract_features_from_list_of_wids([7045, 113, 277, 832, 932], args.num_processes, args.solr_host, args.outfolder, args.lang, args.s3dest)
     #get_wiki_data_from_list(args, [1221, 14316, 23556])
     #extract_features_and_write_to_file(args.num_processes, args.solr_host, args.outfile, args.lang, args.s3dest) 


"""
def main():
    args = get_args()
    features = wikis_to_features(args, get_mainpage_text(args, get_wiki_data(args)))
    flname = args.s3dest if args.s3dest else args.outfile
    with codecs.open(flname, u'w', encoding=u'utf8') as fl:
        for wid, features in features.items():
            line_for_writing = u",".join([wid, u",".join(features)]) + u"\n"
            fl.write(line_for_writing)
    if args.s3dest:
        b = connect_s3().get_bucket(u'nlp-data')
        k = b.get_key(args.s3dest)
        k.set_contents_from_filename(args.s3dest)

def extract_features_and_write_to_file(num_processes=8, solr_host = u"http://search-s10:8983", outfile = u'wiki_data.csv', lang = u'en', s3dest = None):
    
    #essentially a version of main() to be called from within a python script.
    #once finished, main() will become a wrapper for this function.
    #@Author: Isaac
    #:param s3dest: I have no idea
  
    print "WARNING !! Thus function is deprecated, and may be incompatible with other scripts in this module.\
  For instance, \'outfile\' has been replaced with \'outfolder\', in order to account for numeric features.\
 User is reccommended to familiarize him or herself with/modify the code."

    class ArgumentObject:
        def __init__(self, num_processes=8, solr_host = u"http://search-s10:8983", outfile = u'wiki_data.csv', s3dest = None): #lang = u'en'):
            self.num_processes = num_processes
            self.solr_host = solr_host
            self.outfile = outfile
            # self.lang = lang # deprecated in favor of global --Isaac
            self.s3dest = s3dest
    
    args = ArgumentObject(num_processes, solr_host, outfile, s3dest)
    global LANGUAGE
    LANGUAGE = lang # I use a global because otherwise it's a pain to make map_async accept multiple arguments :( --Isaac
    global stops
    stops = stopwords.words(code_to_language_name[LANGUAGE]) # Added --Isaac.  kept global, as Robert had written it, but moved it here so the language could be specified.
    features = wikis_to_features(args, get_mainpage_text(args, get_wiki_data(args)))
    flname = args.s3dest if args.s3dest else args.outfile
    with codecs.open(flname, u'w', encoding=u'utf8') as fl:
        for wid, features in features.items():
            line_for_writing = u",".join([wid, u",".join(features)]) + u"\n"
            fl.write(line_for_writing)
    if args.s3dest:
        b = connect_s3().get_bucket(u'nlp-data')
        k = b.get_key(args.s3dest)
        k.set_contents_from_filename(args.s3dest)
    

"""

if __name__ == u'__main__':
    main()
