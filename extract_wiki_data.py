from argparse import ArgumentParser
from collections import OrderedDict
from text.blob import TextBlob
from nltk.util import bigrams
from multiprocessing import Pool
from traceback import format_exc
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize.regexp import WhitespaceTokenizer
from nltk.corpus import stopwords
from boto import connect_s3
import requests
import codecs
import traceback


stemmer = EnglishStemmer()
tokenizer = WhitespaceTokenizer()
# stops (still global) was originally declared out here.  I moved it into the main method
# so it could be taylored to the language.
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



def get_args():
    ap = ArgumentParser()
    ap.add_argument(u'--num-processes', dest=u"num_processes", default=8, type=int)
    ap.add_argument(u'--solr-host', dest=u"solr_host", default=u"http://search-s10:8983") #TODO: make sure this isn't language specific
    ap.add_argument(u'--outfile', dest=u'outfile', default=u'wiki_data.csv')
    ap.add_argument(u'--s3dest', dest=u's3dest')
    ap.add_argument(u'--lang', dest=u'lang', default = u'en')
    return ap.parse_args()


def get_wiki_data(args):
    """
    Gets wiki data as JSON docs for all English wikis with 50 or more articles (content pages).

    :return: OrderedDict of search docs, id to doc
    :rtype:class:`collections.OrderedDict`
    """
    params = {u'fl': u'id,top_categories_mv_%s,hub_s,top_articles_mv_%s,description_txt,sitename_txt'%(LANGUAGE, LANGUAGE), #TODO: added in args.lang argument where I saw en. Justified?
              u'start': 0,
              u'wt': u'json',
              u'q': u'lang_s:%s AND articles_i:[50 TO *]'%LANGUAGE,
              u'rows': 500}
    data = []
    while True:
        response = requests.get(u'%s/solr/xwiki/select' % args.solr_host, params=params).json()
        data += response[u'response'][u'docs']
        if response[u'response'][u'numFound'] < params[u'rows'] + params[u'start']:
            return OrderedDict([(d[u'id'], d) for d in data])
        params[u'start'] += params[u'rows']


def get_mainpage_text(args, wikis):
    """
    Get mainpage text for each wiki
    :param wikis: our wiki data set
    :type wikis:class:`collections.OrderedDict`
    :return: OrderedDict of search docs, id to doc, with mainpage raw text added
    :rtype:class:`collections.OrderedDict`
    """
    for i in range(0, len(wikis), 100):
        query = u'(%s) AND is_main_page:true' % u' OR ' .join([u"wid:%s" % wid for wid in wikis.keys()[i:i+100]])
        params = {u'wt': u'json',
                  u'start': 0,
                  u'rows': 100,
                  u'q': query,
                  u'fl': u'wid,html_%s'%LANGUAGE}
        response = requests.get(u'%s/solr/main/select' % args.solr_host, params=params).json()
        for result in response[u'response'][u'docs']:
            wikis[str(result[u'wid'])][u'main_page_text'] = result[u'html_%s'%LANGUAGE]

    return wikis


def normalize(wordstring):
    global stemmer, tokenizer, stops
    try:
        return [stemmer.stem(word) for word in tokenizer.tokenize(wordstring.lower())
                if len(word) > 3 and word not in stops]
    except (Exception, IndexError) as e:
        traceback.format_exc()
        print e
        print wordstring
        print tokenizer.tokenize(wordstring.lower())


def wiki_to_feature(wiki):
    """
    Specifically handles a single wiki document
    :param wiki: dict for wiki fields
    :type wiki: dict
    :return: tuple with wiki id and list of feature strings
    :rtype: tuple
    """
    try:
        features = []
        bow = []
        features += [u'ORIGINAL_HUB:%s' % wiki.get(u'hub_s', u'')]
        features += [u'TOP_CAT:%s' % u'_'.join(normalize(c)) for c in wiki.get(u'top_categories_mv_%s'%LANGUAGE, [])] # Added lang --Isaac
        bow += [u"_".join(normalize(c)) for c in wiki.get(u'top_categories_mv_%s'%LANGUAGE, [])] # added lang. --Isaac
        features += [u'TOP_ART:%s' % u"_".join(normalize(a)) for a in wiki.get(u'top_articles_mv_%s'%LANGUAGE, [])] # added lang. --Isaac
        bow += [u"_".join(normalize(a)) for a in wiki.get(u'top_articles_mv_%s'%LANGUAGE, [])]
        desc_ngrams = [u"_".join(n) for grouping in
                       [bigrams(normalize(np))
                       for np in TextBlob(wiki.get(u'description_txt', [u''])[0]).noun_phrases]
                       for n in grouping]
        bow += desc_ngrams
        features += [u'DESC:%s' % d for d in desc_ngrams]
        bow += [u"_".join(b) for b in bigrams(normalize(wiki[u'sitename_txt'][0]))]
        mp_nps = TextBlob(wiki.get(u'main_page_text', u'')).noun_phrases
        bow += [u"_".join(bg) for grouping in [bigrams(normalize(n)) for n in mp_nps] for bg in grouping]
        bow += [u''.join(normalize(w)) for words in [np.split(u" ") for np in mp_nps] for w in words]
        return wiki[u'id'], bow + features
    except Exception as e:
        print e, format_exc()
        raise e


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


def extract_features_and_write_to_file(num_processes=8, solr_host = u"http://search-s10:8983", outfile = u'wiki_data.csv', lang = u'en', s3dest = None):
    """
    essentially a version of main() to be called from within a python script.
    once finished, main() will become a wrapper for this function.
    @Author: Isaac
    :param s3dest: I have no idea
    """

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
    

def main():
    args = get_args()
    extract_features_and_write_to_file(args.num_processes, args.solr_host, args.outfile, args.lang, args.s3dest) 


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
"""

if __name__ == u'__main__':
    main()
