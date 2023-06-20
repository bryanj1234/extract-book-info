import xml.etree.ElementTree as etree
import bz2
import gensim
from gensim.models import word2vec, doc2vec
from gensim.test.utils import datapath, get_tmpfile
#from gensim.corpora import WikiCorpus # Python is case-sensitive duh!
from bryan_genism_wikicorpus import BryanWikiCorpus
import logging
import pickle
import multiprocessing
from pprint import pprint
# import mwparserfromhell
import re



# NOTES #############################################################################################

# See:
    # !!! https://markroxor.github.io/gensim/static/notebooks/doc2vec-wikipedia.html
# Also see:
    # MEMORY MAPPING: https://radimrehurek.com/gensim/models/word2vec.html
    # https://radimrehurek.com/gensim/models/doc2vec.html
    # https://github.com/RaRe-Technologies/gensim/blob/3c3506d51a2caf6b890de3b1b32a8b85f7566ca5/docs/notebooks/doc2vec-IMDB.ipynb
    # https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4
    # https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
    # https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html
    # https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.LineSentence
    # https://mwparserfromhell.readthedocs.io/en/latest/api/mwparserfromhell.html#module-mwparserfromhell.wikicode
    # https://radimrehurek.com/gensim/models/word2vec.html
    # https://radimrehurek.com/gensim/corpora/mmcorpus.html#gensim.corpora.mmcorpus.MmCorpus.serialize
    # https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html

# END NOTES #############################################################################################

### CONFIG ##############################################################################################

# Gensim silently truncates documents to 10000 words, so we need to split up long Wikipedia articles.
max_characters_in_doc_chunk = 30000 # 30000


wikipedia_data_dir = '/home/bryan/Documents/DEV/not-version-controlled/MediaWiki/Wikipedia data'

# Decompressed wikipedia dump file.
# Was originally the file enwiki-20220120-pages-articles-multistream.xml.bz2
# 16,353,051 pages with namespace 0 (ns = 0)?

# FOR REAL
wikipedia_dump_xml_bz2_file_str = wikipedia_data_dir + '/enwiki-20220120-pages-articles-multistream.xml.bz2'

# FOR TESTING. NOT ACTUALLY IN FILES SYSTEM, BUT GENISM WILL FIGURE IT OUT...
#wikipedia_dump_xml_bz2_file_str = "enwiki-latest-pages-articles1.xml-p000000010p000030302-shortened.bz2"

genism_WikiCorpus_wiki_dump_str = wikipedia_data_dir + '/_genism_MmCorpus_wiki_dump/WikiCorpus.pickle'
genism_WikiCorpus_dictionary_str = wikipedia_data_dir + '/_genism_MmCorpus_wiki_dump/WikiCorpus_dictionary.pickle'

genism_Doc2Vec_WikiCorpus_str = wikipedia_data_dir + '/_genism_MmCorpus_wiki_dump/Doc2Vec_WikiCorpus.pickle'

wikipedia_article_titles_pickled_str = wikipedia_data_dir + '/wikipedia_article_titles.pickle'
wikipedia_article_redirects_pickled_str = wikipedia_data_dir + '/wikipedia_article_redirects.pickle'
wikipedia_internal_links_pickled_str = wikipedia_data_dir + '/wikipedia_internal_links.pickle'
wikipedia_internal_link_text_to_link_pickled_str = wikipedia_data_dir + '/wikipedia_internal_link_text_to_link.pickle'
wikipedia_article_category_pickled_str = wikipedia_data_dir + '/wikipedia_article_category.pickle'

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.INFO)


### END CONFIG ##############################################################################################


### JUNK CODE #############################################################################

# wiki_dict = pickle.load(open(genism_WikiCorpus_dictionary_str, "rb"))
# wiki = pickle.load(open(genism_WikiCorpus_wiki_dump_str, "rb"))

# wiki.metadata = True

# ii = -1
# for content, (page_id, title) in wiki.get_texts():
#     ii += 1

#     chunk_start_ind = 0
#     while chunk_start_ind < len(content):
#         content_chunk = content[chunk_start_ind:chunk_start_ind + max_characters_in_doc_chunk]
#         tag_str = title + '_XXX_' + str(page_id) + '_XXX_' + str(chunk_start_ind)
#         print(ii, tag_str, len(content_chunk), content_chunk[:20])
#         chunk_start_ind += max_characters_in_doc_chunk
#     print()
# exit()

# model = doc2vec.Doc2Vec.load(genism_Doc2Vec_WikiCorpus_str)
# print(dir(model))
# print(str(model))
# print(len(model.wv))
# print(model.wv.key_to_index["anarchism"])
# random_word = model.wv.index_to_key[1001]
# print(random_word)
# print(len(model.wv.index_to_key))

# exit()

### END JUNK CODE #############################################################################


###############################################################################################################
### Retrieve tools for using Wikipedia ########################################################################
###############################################################################################################

def get_PV_DBOW_doc2vec_model():
    return doc2vec.Doc2Vec.load(genism_Doc2Vec_WikiCorpus_str)


def get_PV_DBOW_doc2vec_vocab():
    return pickle.load(open(genism_WikiCorpus_dictionary_str, "rb"))

def get_article_titles():
    return pickle.load(open(wikipedia_article_titles_pickled_str, "rb"))


def get_article_redirects():
    return pickle.load(open(wikipedia_article_redirects_pickled_str, "rb"))


def get_internal_links():
    return pickle.load(open(wikipedia_internal_links_pickled_str, "rb"))


def get_internal_link_text_to_link():
    return pickle.load(open(wikipedia_internal_link_text_to_link_pickled_str, "rb"))


def get_article_category():
    return pickle.load(open(wikipedia_article_category_pickled_str, "rb"))

###############################################################################################################
### Helper functions ##########################################################################################
###############################################################################################################

def genism_tokenize(text, doc2vec_wv):
    return [token for token in gensim.utils.tokenize(text, lowercase=True) if token in doc2vec_wv]


def genism_get_most_similar_doc_titles(doc2vec, text_str, topn=10):
    paragraph_vec = doc2vec.infer_vector(genism_tokenize(text_str, doc2vec.wv))
    return doc2vec.dv.most_similar(positive=paragraph_vec, topn=topn)

###############################################################################################################
### Create tools for using Wikipedia ##########################################################################
###############################################################################################################


def genism_make_wiki_corpus():

    # PATH TO DUMP FILE
    path_to_wiki_dump = datapath(wikipedia_dump_xml_bz2_file_str)

    # BUILD
    # With "lower=True", the text is all converted to lower case.
    wiki = BryanWikiCorpus(path_to_wiki_dump, processes=16, lower=True, article_min_tokens=5, token_min_len=2)

    # SERIALIZE DICTIONARY.
    print("Serializing dictionary to", genism_WikiCorpus_dictionary_str)
    pickle.dump(wiki.dictionary, open(genism_WikiCorpus_dictionary_str, "wb"))

    # SERIALIZE ENTIRE WIKIPEDIA MODEL (including texts).
    print("Serializing WikiCorpus to", genism_WikiCorpus_wiki_dump_str)
    pickle.dump(wiki, open(genism_WikiCorpus_wiki_dump_str, "wb"))


# Break up long articles into chunks of length max_characters_in_doc_chunk (or less for the last chunk).
class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title, tag) in self.wiki.get_texts(max_characters_in_doc_chunk=max_characters_in_doc_chunk, chunk_by_page_section=True):
            tag_str = title + '_XXX_' + str(page_id) + ':' + tag
            #print(tag_str)
            yield doc2vec.TaggedDocument([c for c in content], [tag_str])


# ???NO LONGER NEEDED by this version of Gensim???
# # STEP 1) Determin vocab size.
# def determine_vocab_size():
#     # Get wiki corpus
#     wiki = pickle.load(open(genism_WikiCorpus_wiki_dump_str, "rb"))
#     documents = TaggedWikiDocument(wiki)
#     pre = doc2vec.Doc2Vec(min_count=0)
#     pre.scan_vocab(documents)

#     min_count = 5
#     #pre.prepare_vocab(min_count=min_count, dry_run=True)['retain_total']


def create_PV_DBOW_doc2vec_model():
    # Get wiki corpus
    wiki = pickle.load(open(genism_WikiCorpus_wiki_dump_str, "rb"))
    documents = TaggedWikiDocument(wiki)

    # From Top2Vec
    # doc2vec_args = {"vector_size": 300,
    #                    "min_count": min_count,
    #                    "window": 15,
    #                    "sample": 1e-5,
    #                    "negative": 0,
    #                    "hs": 1,
    #                    "epochs": 40,
    #                    "dm": 0,
    #                    "dbow_words": 1}

    # Create the Doc2Vec model.
    num_cpus = 7
    model = doc2vec.Doc2Vec(dm=0, dbow_words=1, hs=1, negative=0, sample=1e-5, vector_size=150, window=15, min_count=5, workers=num_cpus)
    model.build_vocab(documents)
    # Save
    model.save(genism_Doc2Vec_WikiCorpus_str)


def load_and_train_PV_DBOW_doc2vec_model():
    # Load
    model = doc2vec.Doc2Vec.load(genism_Doc2Vec_WikiCorpus_str)

    # Get wiki corpus
    wiki = pickle.load(open(genism_WikiCorpus_wiki_dump_str, "rb"))
    documents = TaggedWikiDocument(wiki)

    # Train
    num_epochs = 10
    num_cpus = 7

    model.workers = num_cpus

    print("########################################################################")
    print("NUMBER OF EPOCHS for training:", num_epochs)
    print("NUMBER OF WORKERS for training:", num_cpus)
    print("########################################################################")

    model.workers=num_cpus
    model.train(documents, total_examples=model.corpus_count, epochs=num_epochs)

    # Save
    model.save(genism_Doc2Vec_WikiCorpus_str)


def inspect_PV_DBOW_doc2vec_model():
    # Load
    model = doc2vec.Doc2Vec.load(genism_Doc2Vec_WikiCorpus_str)

    print(str(model))
    pprint(model.docvecs.most_similar(positive=["Anarchism_XXX_12:19956-39034:3:Origins:0"], topn=20))


def wiki_corpus_get_titles_and_internal_links():

    # FIXME :: Deal with links that put pages in categories.
    #               'Category:Units of measurement'
    #               [[Category:Category name]] or [[Category:Category name|Sortkey]]

    links_re = re.compile(r'\[\[([^\[\]]+)\]\]')
    redirect_re = re.compile(r'#REDIRECT +\[\[([^\[\]]+)\]\]', re.IGNORECASE)

    article_titles = {}                 # Dictionary of Wikipedia article titles
    article_redirects = {}              # Target title by bad title
    internal_links = {}                 # Counts links by article title
    internal_link_text_to_link = {}     # Xref and count of displayed link text to article title
    article_category = {}               # Categories for article by artivle title

    # BRYAN FIXME :: Store results somehow.
    def handle_article(page_title_str, page_ns, page_text):
        if page_ns.isnumeric() and int(page_ns) == 0:           # page_ns = 0 regular article pages...

            # print("==================================================================")
            # print(page_title_str)
            # print("==================================================================")
            #print(links)

            # Titles -------------------------------------------------------------------------------
            if not page_title_str in article_titles:
                article_titles[page_title_str] = True

            # Redirects ----------------------------------------------------------------------------
            redirect_matches = redirect_re.findall(page_text)
            for match in redirect_matches:
                parts = match.strip().split('|')
                link_article = parts[0].strip()
                article_redirects[page_title_str] = link_article

            # Internal links -----------------------------------------------------------------------
            link_matches = links_re.findall(page_text)
            links = []
            link_matches = links_re.findall(page_text)
            for match in link_matches:
                parts = match.split('|')
                link_article = parts[0].strip()
                link_text = parts[1].strip() if len(parts) > 1 else link_article
                links.append((link_article, link_text))

            for link in links:
                link_article = link[0]

                # Internal link vs category placement
                if link_article.startswith('Category:'):        # Category placement
                    category = link_article.replace('Category:', '').strip()
                    if not page_title_str in article_category:
                        article_category[page_title_str] = {}
                    article_category[page_title_str][category] = True
                else:                                           # Internal link
                    link_text = link[1]
                    if not link_article in internal_links:
                        internal_links[link_article] = 0
                    internal_links[link_article] += 1
                    if not link_text in internal_link_text_to_link:
                        internal_link_text_to_link[link_text] = {}
                    if not link_article in internal_link_text_to_link[link_text]:
                        internal_link_text_to_link[link_text][link_article] = 0
                    internal_link_text_to_link[link_text][link_article] += 1

    in_page = False
    page_title_str = False
    page_ns = False
    page_text = False

    count = 0
    for event, elem in etree.iterparse(bz2.open(wikipedia_dump_xml_bz2_file_str), events=("start", "end")):
        tag = elem.tag
        value = elem.text

        # Remove namespace from tag
        prefix, has_namespace, postfix = tag.partition('}')
        if has_namespace:
            tag = postfix  # strip all namespaces

        if tag in ['page', 'title', 'ns', 'text']:
            #print("<<<<", event, tag)

            if tag == 'page':
                if event == 'start':
                    count += 1
                    if count % 100 == 0:
                        print("Page count:", count)
                    # if count > 1000:
                    #     break
                    if not in_page:
                        in_page = True
                    else:
                        print(in_page, page_title_str, page_ns, page_text)
                        assert False, "bad state"
                else: # end
                    if in_page and page_title_str and page_ns and page_text:
                        handle_article(page_title_str, page_ns, page_text)
                        in_page = False
                        page_title_str = False
                        page_ns = False
                        page_text = False
                    else:
                        print(in_page, page_title_str, page_ns, page_text)
                        assert False, "bad state"
                    # Free memory
                    elem.clear()

            elif tag == 'title' and event == 'end':
                page_title_str = value if value is not None else '__NONE__'
                #print("<", "page_title_str:", page_title_str)

            elif tag == 'ns' and event == 'end':
                page_ns = value if value is not None else '__NONE__'
                #print("<", "page_ns:", page_ns)

            elif tag == 'text' and event == 'end':
                page_text = value if value is not None else '__NONE__'
                #print("<", "page_text:", page_text)

    print("ns_0_count:", count)

    # SERIALIZE STUFF
    print("Serializing Wikipedia article_titles to", wikipedia_article_titles_pickled_str)
    pickle.dump(article_titles, open(wikipedia_article_titles_pickled_str, "wb"))
    print("Serializing Wikipedia article_redirects to", wikipedia_article_redirects_pickled_str)
    pickle.dump(article_redirects, open(wikipedia_article_redirects_pickled_str, "wb"))
    print("Serializing Wikipedia internal_links to", wikipedia_internal_links_pickled_str)
    pickle.dump(internal_links, open(wikipedia_internal_links_pickled_str, "wb"))
    print("Serializing Wikipedia internal_link_text_to_link to", wikipedia_internal_link_text_to_link_pickled_str)
    pickle.dump(internal_link_text_to_link, open(wikipedia_internal_link_text_to_link_pickled_str, "wb"))
    print("Serializing Wikipedia article_category to", wikipedia_article_category_pickled_str)
    pickle.dump(article_category, open(wikipedia_article_category_pickled_str, "wb"))


    # UNSERIALIZE TO TEST
    article_titles = pickle.load(open(wikipedia_article_titles_pickled_str, "rb"))
    article_redirects = pickle.load(open(wikipedia_article_redirects_pickled_str, "rb"))
    internal_links = pickle.load(open(wikipedia_internal_links_pickled_str, "rb"))
    internal_link_text_to_link = pickle.load(open(wikipedia_internal_link_text_to_link_pickled_str, "rb"))
    article_category = pickle.load(open(wikipedia_article_category_pickled_str, "rb"))

    print('# article_titles:', len(article_titles))
    print('# article_redirects:', len(article_redirects))
    print('# internal_links:', len(internal_links))
    print('# internal_link_text_to_link:', len(internal_link_text_to_link))
    print('# article_category:', len(article_category))

    # print('\n\n\n#################################################################################')
    # print(article_titles)
    # print('\n\n\n#################################################################################')
    # print(article_redirects)
    # print('\n\n\n#################################################################################')
    # print(internal_links)
    # print('\n\n\n#################################################################################')
    # print(internal_link_text_to_link)
    # print('\n\n\n#################################################################################')
    # print(article_category)



###############################################################################################################
### Create tools for using Wikipedia ##########################################################################
###############################################################################################################

if __name__ == '__main__':

    # --- Wrangle Wikipedia ----------------------------------------------------------------------------

    # STEP 0) Make genism version of wiki dump.
    #genism_make_wiki_corpus()

    # --- Create doc2vec PV-DBOW model from Wikipedia --------------------------------------------------

    # STEP 1) Determine vocab size.
    # ???NO LONGER NEEDED by this version of Gensim???

    # STEP 2) Create and save PV-DBOW model.
    #create_PV_DBOW_doc2vec_model()

    # STEP 3) Load and train PV-DBOW model.
    load_and_train_PV_DBOW_doc2vec_model()

    # STEP 4) Examine resulting model.
    #inspect_PV_DBOW_doc2vec_model()

    # --- Extract article titles and internal link titles from Wikipedia for term extraction -----------

    # STEP 5) For A.T.E. get Wikipedia article titles and internal link titles.
    #wiki_corpus_get_titles_and_internal_links()

