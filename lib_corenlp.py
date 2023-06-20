import stanza
from my_conparse_tree import MyConParseTree
from stanza.server import CoreNLPClient
import graphviz
import spacy, benepar #, coreferee
import os
import pathlib
import pickle

### NOTES ###########################################################################

    ### LIST OF dependency realtions used here and in class MyConParseTree
        # acl:relcl
        # advmod
        # advmod:emph
        # advmod:lmod
        # amod
        # case
        # conj
        # cop
        # csubj
        # nmod
        # nmod:poss
        # nmod:tmod
        # nsubj
        # nsubj:pass
        # obj
        # obl
        # punct
        # relcl
    ### LIST of Spacy dependency relations:
        # acl
        # acomp
        # advcl
        # advmod
        # agent
        # amod
        # appos
        # attr
        # aux
        # auxpass
        # case
        # cc
        # ccomp
        # complm
        # conj
        # cop
        # csubj
        # csubjpass
        # dep
        # det
        # dobj
        # expl
        # hmod
        # hyph
        # infmod
        # intj
        # iobj
        # mark
        # meta
        # neg
        # nmod
        # nn
        # npadvmod
        # nsubj
        # nsubjpass
        # num
        # number
        # obj
        # obl
        # oprd
        # parataxis
        # partmod
        # pcomp
        # pobj
        # poss
        # possessive
        # preconj
        # prep
        # prt
        # punct
        # quantmod
        # rcmod
        # relcl
        # root
        # xcomp

    # Documentation:
    # See https://stanfordnlp.github.io/CoreNLP/corenlp-server.html
    # See https://stanfordnlp.github.io/CoreNLP/cmdline.html
    # See https://stanfordnlp.github.io/CoreNLP/annotators.html
    # See http://www.linguisticsweb.org/doku.php?id=linguisticsweb:tutorials:automaticannotation:stanford_core_nlp
    # See https://stanfordnlp.github.io/CoreNLP/
    # See https://stanfordnlp.github.io/CoreNLP/simple.html
    # See https://stanfordnlp.github.io/CoreNLP/corenlp-server.html
    # See https://colab.research.google.com/github/stanfordnlp/stanza/blob/master/demo/Stanza_CoreNLP_Interface.ipynb
    # See https://stanfordnlp.github.io/stanza/client_setup.html#manual-installation
    # See https://stanfordnlp.github.io/CoreNLP/caseless.html

    # Starting on command line
    # java -cp "/home/bryan/stanza_corenlp/*" -mx8g edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -annotators tokenize,ssplit,pos,lemma,ner,parse,depparse,coref
    # java -cp "/home/bryan/stanza_corenlp/*" -mx8g edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -annotators tokenize,ssplit,pos,lemma,parse,coref

def core_nlp_get_corenlp_and_stanza():
    # # NOTE! Can't use "~" for home here. Should give a full path.
    # corenlp_dir = '/home/bryan/stanza_corenlp'
    # #stanza.install_corenlp(dir=corenlp_dir) # ONly need to run once.
    # os.environ['CORENLP_HOME'] = corenlp_dir

    # Connect to CoreNLP Java service running in background
    #nlp_client = CoreNLPClient(start_server=stanza.server.StartServer.DONT_START, endpoint='http://localhost:9000')

    # Make a new CoreNLP Java process
    nlp_client = CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'])

    nlp_client.stanza_nlp_client = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency,depparse,lemma', tokenize_pretokenized=True)

    return nlp_client

def get_spacy(model_picked_file_str=False):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    spacy_client = False
    if model_picked_file_str:
        spacy_client = pickle.load(open(model_picked_file_str, "rb"))
    else:
        spacy_client = spacy.load("en_core_web_trf")
        spacy_client.add_pipe('benepar', config={'model': 'benepar_en3'})
        spacy_client.add_pipe('coreferee')

    return spacy_client

class NLPClientLoader:
    def __init__(self, nlp_package_str, model_picked_file_str=False, reload_every=False):
        valid_package_names = ['SPACY', 'STANZA_WITH_CORENLP']
        assert nlp_package_str in valid_package_names, "Bad value for nlp_package_str, needs to be in " + str(valid_package_names)
        self.nlp_package_str = nlp_package_str
        self.reload_every = reload_every
        self.num_loads = -1
        self.nlp_client = False
        self.model_picked_file_str = model_picked_file_str

    def load_nlp_client(self):
        self.num_loads += 1

        if self.reload_every is False or (self.num_loads % self.reload_every == 0):
            print("Getting " + self.nlp_package_str + " NLP client(s)...")
            if self.nlp_package_str == 'STANZA_WITH_CORENLP':
                self.nlp_client = core_nlp_get_corenlp_and_stanza()
            elif self.nlp_package_str == 'SPACY':
                self.nlp_client = get_spacy(model_picked_file_str=self.model_picked_file_str)
            print("... Done getting " + self.nlp_package_str + " NLP client(s)...")

        return self.nlp_client

#-----------------------------------------------------------------------------------------------


def stanza_create_idx_words(dependencies):
    idx_words = {}
    for dep_word in dependencies:
        word_id = dep_word.id
        word = {'id': word_id,
                'text': dep_word.text,
                'head': dep_word.head,
                'pos': dep_word.xpos,
                'deprel': dep_word.deprel}
        idx_words[word_id] = word

    return idx_words


def get_stanza_constituency_tree_and_idx_words(stanza_nlp_client, idx_words):
    word_tokens = [idx_words[word_id]['text'] for word_id in idx_words]
    stanza_doc = stanza_nlp_client([word_tokens]) # Run Stanza on pre-tokenized sentence.
    stanza_sentence = stanza_doc.sentences[0]
    stanza_dependencies = stanza_sentence.words
    idx_words_stanza = stanza_create_idx_words(stanza_dependencies)
    stanza_parse_tree = MyConParseTree.get_from_stanza_contree(stanza_sentence.constituency)
    stanza_parse_tree.add_idx_words(idx_words_stanza)

    return stanza_parse_tree


def get_spacy_constituency_tree_and_idx_words(spacy_client, orig_idx_words):
    words = [orig_idx_words[word_id]['text'] for word_id in orig_idx_words]

    doc = spacy.tokens.doc.Doc(
            spacy_client.vocab, words=words)

    for name, proc in spacy_client.pipeline:
        doc = proc(doc)

    # Get idx_words, again
    sent = list(doc.sents)[0]
    sent_tokens = list(token for token in sent)
    idx_words = spacy_create_idx_words(sent_tokens)

    # Replace lemma and doc_token_index and from original idx_words...
    # Also make XREF of original doc token id to word.
    tmp_xref_doc_token_num_to_word_info = {}
    for word_id in idx_words:
        idx_words[word_id]['lemma'] = orig_idx_words[word_id]['lemma']
        idx_words[word_id]['doc_token_index'] = orig_idx_words[word_id]['doc_token_index']
        tmp_xref_doc_token_num_to_word_info[idx_words[word_id]['doc_token_index']] = idx_words[word_id]

    # Get constituency tree
    spacy_parse_tree = MyConParseTree.get_from_spacy_contree(idx_words, sent)

    return spacy_parse_tree, idx_words, tmp_xref_doc_token_num_to_word_info

def core_nlp_create_idx_words(tokens, dependencies):
    idx_words = {}
    for token in tokens:
        word_id = token.endIndex
        idx_words[word_id] = {'id': word_id,
                              'text': token.word.strip(),
                              'head': 0,
                              'pos': token.pos,
                              'deprel': 'root'}

    for edge in dependencies.edge:
        word_id = edge.target
        head_id = edge.source
        deprel = edge.dep
        idx_words[word_id]['head'] = head_id
        idx_words[word_id]['deprel'] = deprel

    return idx_words

def spacy_create_idx_words(sent_tokens):
    idx_words = {}

    xref_token_i_to_word_id = {}

    # Do first pass
    ii = 0
    for token in sent_tokens:
        ii += 1
        word_id = ii
        xref_token_i_to_word_id[token.i] = word_id
        word = {'id': word_id,
                'doc_token_index':token.i,
                'text': token.text,
                'head': 0,                  # Set in second pass.
                'pos': token.tag_,
                'deprel': token.dep_.lower(),
                'lemma':token.lemma_}
        idx_words[word_id] = word
    # Do second pass
    ii = 0
    for token in sent_tokens:
        ii += 1
        head_id = ii
        for dep_child_token in token.children:
            word_id = xref_token_i_to_word_id[dep_child_token.i]
            idx_words[word_id]['head'] = head_id

    return idx_words

def get_sentence_trees_from_text_Stanza_and_CoreNLP(text, corenlp_client):

    # Have CoreNLP break text into sentences, and tokenize each sentence.
    annotations = corenlp_client.annotate(text)

    sentence_trees = []
    for sentence in annotations.sentence:

        # Get word tokens back from CoreNLP.
        idx_words = core_nlp_create_idx_words(sentence.token, sentence.basicDependencies)

        orig_text = ' '.join([idx_words[word_id]['text'] for word_id in idx_words])

        # Pre-process/re-tokenize idx_words: Concatenate compound verbs, nouns, adjectives into single tokens.
        idx_words = preprocess_and_retokenize_idx_words(idx_words)

        # Have Stanza parse the preprocessed word list, and get a constituency tree back.
        parse_tree = get_stanza_constituency_tree_and_idx_words(corenlp_client.stanza_nlp_client, idx_words)

        sentence_trees.append({'idx_words':idx_words, 'parse_tree':parse_tree, 'text':orig_text})

    return sentence_trees


def get_sentence_trees_from_text_Spacy(text, spacy_client):
    doc = spacy_client(text)

    xref_doc_token_num_to_word_info = {}

    sentence_list = []
    for sent in doc.sents:
        sent_tokens = list(token for token in sent)
        # Get idx_words
        idx_words = spacy_create_idx_words(sent_tokens)

        orig_text = ' '.join([idx_words[word_id]['text'] for word_id in idx_words])

        # Pre-process/re-tokenize idx_words: Concatenate compound verbs, nouns, adjectives into single tokens.
        idx_words = preprocess_and_retokenize_idx_words(idx_words)

        # Get parse tree
        parse_tree, idx_words, tmp_xref_doc_token_num_to_word_info = get_spacy_constituency_tree_and_idx_words(spacy_client, idx_words)

        # Merge token XREFs
        xref_doc_token_num_to_word_info = {**xref_doc_token_num_to_word_info, **tmp_xref_doc_token_num_to_word_info}

        # Append to list of sentences to process below.
        sentence_list.append((parse_tree, idx_words, orig_text))

    sentence_trees = []
    for sentence_rec in sentence_list:
        parse_tree = sentence_rec[0]
        idx_words = sentence_rec[1]
        orig_text = sentence_rec[2]

        # Coreference resolution
        for word_id in idx_words:
            word = idx_words[word_id]
            doc_token_index = word['doc_token_index']
            if doc_token_index is not False:

                # Resolve step 1
                #coref_res_doc_token_index = coref_chain[0][0]
                res_tokens = doc._.coref_chains.resolve(doc[doc_token_index])

                if res_tokens is not None and len(res_tokens) > 0:

                    # FIXME :: There can be more than one ( they -> [peter, wife] )
                    coref_res_doc_token_index = res_tokens[0].i
                    all_coref_res_doc_token_index = [res_token.i for res_token in res_tokens]

                    # Resolve step 2
                    coref_resolved_text = xref_doc_token_num_to_word_info[coref_res_doc_token_index]['text']
                    coref_resolved_lemma = xref_doc_token_num_to_word_info[coref_res_doc_token_index]['lemma']

                    # Add into word
                    word['coref_res_doc_token_index'] = coref_res_doc_token_index
                    word['all_coref_res_doc_token_index'] = all_coref_res_doc_token_index
                    word['coref_resolved_text'] = coref_resolved_text
                    word['coref_resolved_lemma'] = coref_resolved_lemma



        # Add idx_words
        parse_tree.add_idx_words(idx_words)

        sentence_trees.append({'idx_words':idx_words, 'parse_tree':parse_tree, 'text':orig_text})

    return sentence_trees


def get_sentence_trees(text, nlp_client, nlp_package_str):

    if nlp_package_str == 'STANZA_WITH_CORENLP':
        sentence_trees = get_sentence_trees_from_text_Stanza_and_CoreNLP(text, nlp_client)
        return sentence_trees
    elif nlp_package_str == 'SPACY':
        sentence_trees = get_sentence_trees_from_text_Spacy(text, nlp_client)
        return sentence_trees




#----------------------------------------------------------------------------------------------


def core_nlp_print_dependency_tree(dep_tree, level=0):
    print("\t" * level, dep_tree['id'], dep_tree['text'], dep_tree['head'], dep_tree['pos'], dep_tree['deprel'])
    for child in dep_tree['children']:
        core_nlp_print_dependency_tree(child, level=level + 1)


#----------------------------------------------------------------------------------------------

def tweak_copulas(idx_words):
    idx_cop_and_head_ids_seen = {}
    for word_id in idx_words:
        if word_id in idx_cop_and_head_ids_seen:
            continue
        word = idx_words[word_id]
        if word['deprel'] == 'cop':  # Found copula!
            head_id = word['head']
            idx_cop_and_head_ids_seen[word_id] = True
            idx_cop_and_head_ids_seen[head_id] = True
            cop_head_word = idx_words[head_id]
            word['head'] = cop_head_word['head']
            word['deprel'] = cop_head_word['deprel']
            cop_head_word['head'] = word_id
            cop_head_word['deprel'] = 'cop'
            for tmp_word_id in idx_words:
                if tmp_word_id in [word_id, head_id]:
                    continue
                tmp_word = idx_words[tmp_word_id]
                if tmp_word['deprel'] in ['nsubj', 'nsubj:pass', 'csubj']:
                    if tmp_word['head'] == head_id:
                        tmp_word['head'] = word_id


def swap_word_ids_in_idx_words(idx_words, word_id_1, word_id_2):
    # Need to swap word_id_1 and word_id_2
    word_1 = idx_words[word_id_1]
    word_2 = idx_words[word_id_2]
    idx_words[word_id_1] = word_2
    idx_words[word_id_2] = word_1
    # Deal with 'id' and 'head' for all thw words.
    for tmp_word_id in idx_words:
        tmp_word = idx_words[tmp_word_id]
        if tmp_word['id'] == word_id_1:
            tmp_word['id'] = word_id_2
        elif tmp_word['id'] == word_id_2:
            tmp_word['id'] = word_id_1
        if tmp_word['head'] == word_id_1:
            tmp_word['head'] = word_id_2
        elif tmp_word['head'] == word_id_2:
            tmp_word['head'] = word_id_1


# Deal with VERB ADVERB VERB ("we will now show")
def move_trapped_adverb_before_verbs(idx_words):
    bool_was_changed = True
    while bool_was_changed:
        bool_was_changed = False
        word_ids = sorted(idx_words.keys())
        for ii in range(len(word_ids) - 2):
            word_id_1 = word_ids[ii]
            word_id_2 = word_ids[ii + 1]
            word_id_3 = word_ids[ii + 2]
            word_1 = idx_words[word_id_1]
            word_2 = idx_words[word_id_2]
            word_3 = idx_words[word_id_3]

            if word_1['pos'] in ['MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] and not word_1['deprel'] in ['amod']:
                if word_2['pos'] in ['RB', 'RBR', 'RBS']:
                    if word_3['pos'] in ['MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] and not word_3['deprel'] in ['amod']:
                        # Need to swap word_id_1 and word_id_2
                        swap_word_ids_in_idx_words(idx_words, word_id_1, word_id_2)
                        bool_was_changed = True
                        break

def verify_no_empty_words_idx_words(idx_words):
    for word_id in idx_words:
        assert idx_words[word_id]['text'] != ''

# Concatenates all the text into the last word in the group.
# Deals with the ids of the other words then removes them from idx_words.
# NOTE: There will probably be problems if one of the to-be-removed words is the head of the final word.
#   The way to deal with this is to extract the new words from idx_words and run them through Stanza/CoreNLP as pretokenized text, if possible.
#   So, basically just call this function as a preprocessing step and then rerun the NLP stuff on pre-tokenized text.
def concatenate_words_in_idx_words(idx_words, word_group):
    final_word = word_group[len(word_group) - 1]
    final_word_id = final_word['id']
    text_parts = []
    lemma_parts = []
    for ii in range(len(word_group) - 1):
        word = word_group[ii]
        word_id = word['id']
        text_parts.append(word['text'].strip())
        lemma_parts.append(word['lemma'].strip())
        # Remove word
        idx_words.pop(word_id)
        # Remap all other words
        for tmp_word_id in idx_words:
            tmp_word = idx_words[tmp_word_id]
            if tmp_word['head'] == word_id:
                tmp_word['head'] = final_word_id
    text_parts.append(final_word['text'].strip())
    lemma_parts.append(final_word['lemma'].strip())
    final_word['text'] = ' '.join(text_parts)
    final_word['lemma'] = ' '.join(lemma_parts)

    # Sometimes this results in empty text and lemma, for example when the final token is a newline.
    # SpaCy will crash on this, so change it to a newline.
    if final_word['text'] == '':
        final_word['text'] = "\n"
    if final_word['lemma'] == '':
        final_word['lemma'] = "\n"

    # Validate
    verify_no_empty_words_idx_words(idx_words)

# Include adjective modifiers
def get_compound_nouns(idx_words):
    word_ids = reversed(sorted(idx_words.keys()))
    word_groups = []
    cur_group = []

    last_pass_hyphen = False
    for word_id in word_ids:
        word = idx_words[word_id]
        if word['pos'] in ['NN', 'NNS', 'NNP', 'NNPS']:
            cur_group.append(word)
            last_pass_hyphen = False
        elif cur_group != [] and word['pos'] in ['JJ', 'JJR', 'JJS', 'FW']:
            cur_group.append(word)
            last_pass_hyphen = False
        elif cur_group != [] and (word['text'] == '–' or word['pos'] in ['HYPH']):
            cur_group.append(word)
            last_pass_hyphen = True
        elif last_pass_hyphen:      # Who cares what the word before the hyphen is?
            cur_group.append(word)
            last_pass_hyphen = False
        elif cur_group != []:
            last_pass_hyphen = False
            word_groups.append(list(reversed(cur_group)))
            cur_group = []

    if cur_group != []:
        word_groups.append(list(reversed(cur_group)))
        cur_group = []

    return word_groups


def get_compound_verbs(idx_words):
    word_ids = reversed(sorted(idx_words.keys()))
    word_groups = []
    cur_group = []
    for word_id in word_ids:
        word = idx_words[word_id]
        if word['pos'] in ['MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] and not word['deprel'] in ['amod']:
            cur_group.append(word)
        elif cur_group != []:
            word_groups.append(list(reversed(cur_group)))
            cur_group = []
    if cur_group != []:
        word_groups.append(list(reversed(cur_group)))
        cur_group = []

    return word_groups

def get_compound_adjectives(idx_words):
    word_ids = reversed(sorted(idx_words.keys()))
    word_groups = []
    cur_group = []
    for word_id in word_ids:
        word = idx_words[word_id]
        if word['pos'] in ['JJ', 'JJR', 'JJS'] or word['deprel'] in ['amod']:
            cur_group.append(word)
        elif cur_group != []:
            word_groups.append(list(reversed(cur_group)))
            cur_group = []
    if cur_group != []:
        word_groups.append(list(reversed(cur_group)))
        cur_group = []

    return word_groups

def get_compound_prepositions(idx_words):
    word_ids = reversed(sorted(idx_words.keys()))
    word_groups = []
    cur_group = []
    for word_id in word_ids:
        word = idx_words[word_id]
        if word['pos'] in ['IN']:
            cur_group.append(word)
        elif cur_group != []:
            word_groups.append(list(reversed(cur_group)))
            cur_group = []
    if cur_group != []:
        word_groups.append(list(reversed(cur_group)))
        cur_group = []

    return word_groups

# Does not mutate!
def make_indices_one_based_and_sequential(idx_words):

    xref_old_new = {}
    new_id = 0
    for word_id in idx_words:
        new_id += 1
        xref_old_new[word_id] = new_id

    new_idx_words = {}
    for word_id in idx_words:
        word = idx_words[word_id]
        new_id = xref_old_new[word_id]
        word['id'] = new_id
        if word['head'] > 0:
            word['head'] = xref_old_new[word['head']]
        new_idx_words[new_id] = word

    return new_idx_words




# Does not mutate!
def preprocess_and_retokenize_idx_words(idx_words):

    # Rearrange copulas so they look more like regular verbs.
    tweak_copulas(idx_words)
    verify_no_empty_words_idx_words(idx_words)

    # Deal with VERB ADVERB VERB ("we will now show"), so can get compund verbs.
    move_trapped_adverb_before_verbs(idx_words)
    verify_no_empty_words_idx_words(idx_words)

    # Concatenate compound nouns and verbs...
    word_groups = get_compound_nouns(idx_words)
    for word_group in word_groups:
       concatenate_words_in_idx_words(idx_words, word_group)
    word_groups = get_compound_verbs(idx_words)
    for word_group in word_groups:
        concatenate_words_in_idx_words(idx_words, word_group)
    word_groups = get_compound_adjectives(idx_words)
    for word_group in word_groups:
        concatenate_words_in_idx_words(idx_words, word_group)
    word_groups = get_compound_prepositions(idx_words)
    for word_group in word_groups:
        concatenate_words_in_idx_words(idx_words, word_group)

    idx_words = make_indices_one_based_and_sequential(idx_words)


    return idx_words


# ----------------------------------------------------------------------------------------------


def label_is_clause_level(label):
    return label in ['S', 'SBAR', 'SBARQ', 'SINV', 'SQ']


def label_is_phrase_level(label):
    return label in ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX',
                     'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X']


def is_wh_word(parse_tree):
    if parse_tree.word_info['pos'] in ['WDT', 'WP', 'WP$'] and parse_tree.word_info['deprel'] in ['nsubj', 'nmod:poss', 'poss'
                                                                                                  'obl', 'obj', 'dobj', 'pobj'] \
            or parse_tree.word_info['pos'] == 'WRB' and parse_tree.word_info['deprel'] == 'advmod':

        return True

    return False

def is_adjective_like(parse_tree):
    if parse_tree.label in ['ADJP', 'JJ', 'JJR', 'JJS'] \
             or parse_tree.word_info and parse_tree.word_info['deprel'] in ['amod']:
        return True

    return False


def is_noun_like(parse_tree):
    if parse_tree.label in ['WHNP', 'WP', 'NP', 'NN', 'NNS', 'NNP', 'NNPS', 'PRP'] \
            or parse_tree.word_info and parse_tree.word_info['deprel'] in ['nsubj', 'nsubj:pass']:
        return True

    return False


def is_verb_like(parse_tree):
    return parse_tree.label in ['VP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] \
           and not (parse_tree.word_info and parse_tree.word_info['deprel'] in ['amod'])


# ----------------------------------------------------------------------------------------------

def get_graph_node_for_subtree(parse_tree):
    word_info = None
    is_noun = False
    doc_token_index = False
    coref_res_doc_token_index = False
    all_coref_res_doc_token_index = False
    coref_resolved_text = False
    coref_resolved_lemma = False
    label = parse_tree.label
    UID = parse_tree.UID
    graph_pos = parse_tree.graph_pos
    if parse_tree.is_preterminal():
        word_info = parse_tree.word_info
        label = parse_tree.children[0].label + ' (' + parse_tree.label + '/' + word_info['deprel'] + ')'

        doc_token_index = word_info['doc_token_index']
        coref_res_doc_token_index = word_info['coref_res_doc_token_index']
        all_coref_res_doc_token_index = word_info['all_coref_res_doc_token_index']
        coref_resolved_text = word_info['coref_resolved_text']
        coref_resolved_lemma = word_info['coref_resolved_lemma']
        if coref_res_doc_token_index is not False and coref_resolved_text.lower() != word_info['text'].lower():
            label = label + "\n" + "(" + coref_resolved_text + ")"

        is_noun = parse_tree.label in ['NN', 'NNS', 'NNP', 'NNPS']
        if is_noun:
            if word_info['lemma'].lower() != word_info['text'].lower():
                label = label + "\n" + "<" + word_info['lemma'] + ">"


    graph_node = {'UID': UID,
                  'word_info':word_info,
                  'graph_label':label,
                  'graph_pos':graph_pos,
                  'is_noun':is_noun,
                  'doc_token_index':doc_token_index,
                  'coref_res_doc_token_index':coref_res_doc_token_index,
                  'all_coref_res_doc_token_index':all_coref_res_doc_token_index,
                  'coref_resolved_text':coref_resolved_text,
                  'coref_resolved_lemma':coref_resolved_lemma}

    return graph_node

# ----------------------------------------------------------------------------------------------

def get_idx_main_uids_and_idx_word_uids(parse_tree, idx_preterms_by_word_id):
    idx_main_UIDs = {}
    idx_word_UIDs = {}

    idx_deprel_by_unresolved_head_ids_highest = parse_tree.get_idx_deprel_by_unresolved_head_ids_highest()

    idx_clause_level_main_word_ids = {}
    for idx in idx_deprel_by_unresolved_head_ids_highest:
        head_id = idx_deprel_by_unresolved_head_ids_highest[idx][0]
        label = idx_deprel_by_unresolved_head_ids_highest[idx][2]
        word_id = idx_deprel_by_unresolved_head_ids_highest[idx][5][1]
        if label_is_clause_level(label):
            idx_clause_level_main_word_ids[word_id] = True
            idx_clause_level_main_word_ids[head_id] = True

    for word_id in idx_clause_level_main_word_ids:
        UID = idx_preterms_by_word_id[word_id].UID
        word = idx_preterms_by_word_id[word_id]
        if is_noun_like(word) or is_verb_like(word):
            idx_main_UIDs[UID] = 'main_noun' if is_noun_like(word) else 'main_verb'
        elif is_adjective_like(word):
            idx_main_UIDs[UID] = 'main_adjective'
        head_id = idx_preterms_by_word_id[word_id].word_info['head']
        if head_id > 0:
            UID = idx_preterms_by_word_id[head_id].UID
            if is_noun_like(idx_preterms_by_word_id[head_id]) or is_verb_like(idx_preterms_by_word_id[head_id]):
                idx_main_UIDs[UID] = 'main_noun' if is_noun_like(idx_preterms_by_word_id[head_id]) else 'main_verb'
            elif is_adjective_like(idx_preterms_by_word_id[head_id]):
                idx_main_UIDs[UID] = 'main_adjective'

    for word_id in idx_preterms_by_word_id:
        word = idx_preterms_by_word_id[word_id]
        idx_word_UIDs[word.UID] = True

        if not word.UID in idx_main_UIDs:
            if word.word_info['head'] == 0:
                UID = word.UID
                idx_main_UIDs[UID] = 'root'
            elif word.word_info['deprel'] in ['nsubj', 'nsubj:pass', 'csubj']:
                UID = word.UID
                idx_main_UIDs[UID] = 'subject'
            elif word.word_info['head'] > 0:
                head_id = word.word_info['head']
                sub_uid = word.subsentence_UID
                head_sub_uid = idx_preterms_by_word_id[head_id].subsentence_UID
                if sub_uid != head_sub_uid:
                    if is_noun_like(word) or is_verb_like(word):
                        idx_main_UIDs[word.UID] = 'head_different_subsentence'

        if not word.UID in idx_main_UIDs:
            if is_noun_like(word) or is_verb_like(word):
                idx_main_UIDs[word.UID] = 'noun' if is_noun_like(word) else 'verb'
            elif is_adjective_like(word):
                idx_main_UIDs[word.UID] = 'adjective'

    return idx_main_UIDs, idx_word_UIDs


# Resolve WH-pronouns
# Not really very smart. Just goes to the first noun/pronoun nefore it.
def add_wh_word_resolution_edges(idx_wh_word_edges, idx_edges, parse_tree):
    # Get all nodes in idx_edges. Remove "extra" from edges.
    idx_nodes = {}
    for edge in idx_edges:
        idx_nodes[edge[0]] = True
        idx_nodes[edge[1]] = True

    idx_ui_ds_to_subtrees = parse_tree.idx_UIDs_to_subtrees
    preterminals = list(parse_tree.preterminals())
    preterminals = sorted(preterminals, key=lambda rec: (-1) * rec.word_info['id'])

    idx_preterms_by_word_id = {}
    for pt_tree in parse_tree.preterminals():
        idx_preterms_by_word_id[pt_tree.word_info['id']] = pt_tree

    for ii in range(len(preterminals)):
        pt_node = preterminals[ii]
        # print(pt_node.word_info)
        if is_wh_word(pt_node):
            # Make sure should actually try to resolve it!
            smallest_containing_sentence_tree = idx_ui_ds_to_subtrees[pt_node.subsentence_UID]
            idx_deprel_by_unresolved_head_ids = smallest_containing_sentence_tree.idx_deprel_by_unresolved_head_ids
            bool_should_resolve = False
            resolve_to_id = False
            for head_id in idx_deprel_by_unresolved_head_ids:
                for deprel in idx_deprel_by_unresolved_head_ids[head_id]:
                    if deprel in ['acl:relcl', 'relcl']:
                        bool_should_resolve = True
                        resolve_to_id = head_id
                        break
            if not bool_should_resolve:
                continue

            # First version: closest noun before WH-word.
            alt_node_1 = False
            for jj in range(ii + 1, len(preterminals)):
                alt_node = preterminals[jj]
                # print("\t", alt_node.word_info)
                if is_noun_like(alt_node):
                    alt_node_1 = alt_node
                    break

            # Second version. Look at head of head.
            alt_node_2 = False
            head_id = pt_node.word_info['head']
            if head_id > 0 and head_id in idx_preterms_by_word_id:
                head_id = idx_preterms_by_word_id[head_id].word_info['head']
                if head_id > 0 and head_id in idx_preterms_by_word_id:
                    alt_node_2 = idx_preterms_by_word_id[head_id]

            # Third version. Use resolve_to_id.
            alt_node_3 = False
            if resolve_to_id > 0 and resolve_to_id in idx_preterms_by_word_id:
                alt_node_3 = idx_preterms_by_word_id[resolve_to_id]

            # Make edges
            if alt_node_1:
                idx_wh_word_edges[(alt_node_1.UID, pt_node.UID, 'WH-res v1')] = True
            if alt_node_2:
                idx_wh_word_edges[(alt_node_2.UID, pt_node.UID, 'WH-res v2')] = True
            if alt_node_3:
                idx_wh_word_edges[(alt_node_3.UID, pt_node.UID, 'WH-res v3')] = True


def get_copula_edge(parse_tree, idx_preterms_by_word_id):
    # copula -> copula.head = word_1
    # word_2 -> word_2.head = word_1
    if parse_tree.word_info['deprel'] == 'cop':
        word_1_id = parse_tree.word_info['head']
        word_1 = idx_preterms_by_word_id[word_1_id]
        # Find a noun subject whose head is word_1_id.
        word_2 = False
        for word_2_id in idx_preterms_by_word_id:
            tmp_word_2 = idx_preterms_by_word_id[word_2_id]
            if tmp_word_2.word_info['head'] == word_1_id and tmp_word_2.word_info['deprel'] in ['nsubj', 'csubj']:
                word_2 = tmp_word_2
                break

        if word_2:
            extra = "COPULA relationship"
            # print("get_copula_edge()", word_1.UID, word_2.UID, extra)
            return word_2.UID, word_1.UID, extra

    # Default
    return False


def add_all_dependency_edges(idx_dep_tree_words, idx_edges, idx_target_edges, idx_preterms_by_word_id):
    for word_id in idx_preterms_by_word_id:
        pt_node = idx_preterms_by_word_id[word_id]

        head_id = pt_node.word_info['head']

        if head_id > 0:

            extra = pt_node.label + ":" + pt_node.word_info['deprel']

            UID = pt_node.UID
            target_pt_node = idx_preterms_by_word_id[head_id]
            target_uid = target_pt_node.UID

            if idx_dep_tree_words[word_id]['deprel'] == 'conj':
                extra = 'CONJ'
                idx_edges[(UID, target_uid, extra)] = True
                idx_target_edges[(UID, target_uid, extra)] = True
                idx_edges[(target_uid, UID, extra)] = True
                idx_target_edges[(target_uid, UID, extra)] = True
            else:
                idx_edges[(target_uid, UID, extra)] = True
                idx_target_edges[(target_uid, UID, extra)] = True


def add_edges_for_word_order(idx_edges, idx_orig_order_edges, idx_preterms_by_word_id):
    word_ids = list(sorted(idx_preterms_by_word_id.keys()))
    for ii in range(len(word_ids)):
        if ii > 0:
            word_a_id = word_ids[ii - 1]
            word_b_id = word_ids[ii]
            #if idx_preterms_by_word_id[word_a_id].subsentence_UID == idx_preterms_by_word_id[word_b_id].subsentence_UID:
            word_a_uid = idx_preterms_by_word_id[word_a_id].UID
            word_b_uid = idx_preterms_by_word_id[word_b_id].UID
            idx_edges[(word_a_uid, word_b_uid, 'original order')] = True
            idx_orig_order_edges[(word_a_uid, word_b_uid, 'original order')] = True


# Recursive.
# Basically the only difference between this and the original constituency tree is that
# sometimes a sibling will link to another sibling instead of its parent.
def worker_get_graph_from_constituency_tree(parse_tree,
                                            idx_edges,
                                            idx_copula_edges, idx_target_edges, idx_preterms_by_word_id,
                                            idx_dep_tree_words):
    UID_A = parse_tree.UID

    for child_tree in parse_tree.children:

        # Add copula links.
        if child_tree.is_preterminal():
            copula_edge = get_copula_edge(child_tree, idx_preterms_by_word_id)
            if copula_edge:
                idx_edges[copula_edge] = True
                idx_copula_edges[copula_edge] = True

        # Also add in link to parent if need to.
        if label_is_clause_level(child_tree.label) or label_is_phrase_level(child_tree.label) or parse_tree.label =='ROOT':
            UID_B = child_tree.UID
            extra = 'parent'
            idx_edges[(UID_A, UID_B, extra)] = True
        elif child_tree.is_preterminal():
            UID_B = child_tree.UID
            extra = 'parent'
            idx_edges[(UID_A, UID_B, extra)] = True

        # Recurse
        if not child_tree.is_preterminal():
            worker_get_graph_from_constituency_tree(child_tree,
                                                    idx_edges,
                                                    idx_copula_edges, idx_target_edges, idx_preterms_by_word_id,
                                                    idx_dep_tree_words)


def get_graph_from_constituency_tree(parse_tree, idx_dep_tree_words):
    idx_uids_to_subtrees = parse_tree.idx_UIDs_to_subtrees

    idx_preterms_by_word_id = {}
    for pt_tree in parse_tree.preterminals():
        idx_preterms_by_word_id[pt_tree.word_info['id']] = pt_tree

    idx_edges = {}
    idx_copula_edges = {}
    idx_target_edges = {}
    idx_orig_order_edges = {}
    worker_get_graph_from_constituency_tree(parse_tree,
                                            idx_edges,
                                            idx_copula_edges, idx_target_edges, idx_preterms_by_word_id,
                                            idx_dep_tree_words)

    add_all_dependency_edges(idx_dep_tree_words, idx_edges, idx_target_edges, idx_preterms_by_word_id)
    add_edges_for_word_order(idx_edges, idx_orig_order_edges, idx_preterms_by_word_id)

    # Get WH-word resolution edges. (Ex: "the dog which")
    idx_wh_word_edges = {}
    add_wh_word_resolution_edges(idx_wh_word_edges, idx_edges, parse_tree)

    # Extract nodes
    idx_nodes = {}
    for edge in idx_edges.keys():
        node_A_UID = edge[0]
        node_B_UID = edge[1]

        node_a = get_graph_node_for_subtree(idx_uids_to_subtrees[node_A_UID])
        node_b = get_graph_node_for_subtree(idx_uids_to_subtrees[node_B_UID])

        idx_nodes[node_A_UID] = node_a
        idx_nodes[node_B_UID] = node_b

    # Get important-looking nodes so we can color-code them.
    # Get UIDs for words so can color code them.
    idx_main_UIDs, idx_word_UIDs = get_idx_main_uids_and_idx_word_uids(parse_tree, idx_preterms_by_word_id)

    return idx_nodes, idx_edges, idx_word_UIDs, idx_main_UIDs, idx_copula_edges, idx_target_edges, idx_orig_order_edges, \
           idx_wh_word_edges

 # Add edges between noun nodes that have the same word.
def add_edges_between_identical_nouns(idx_nodes, idx_edges, idx_identical_noun_edges):
    idx_noun_nodes_by_noun = {}
    for node_id in idx_nodes:
        graph_node = idx_nodes[node_id]
        if graph_node['is_noun']:
            noun_lemma = graph_node['word_info']['lemma']
            if not noun_lemma in idx_noun_nodes_by_noun:
                idx_noun_nodes_by_noun[noun_lemma] = []
            idx_noun_nodes_by_noun[noun_lemma].append(node_id)
    for noun_lemma in idx_noun_nodes_by_noun:
        for ii in range(len(idx_noun_nodes_by_noun[noun_lemma])):
            for jj in range(ii + 1, len(idx_noun_nodes_by_noun[noun_lemma])):
                UID_1 = idx_noun_nodes_by_noun[noun_lemma][ii]
                UID_2 = idx_noun_nodes_by_noun[noun_lemma][jj]
                edge_1 = (UID_1, UID_2, 'identical_nouns')
                edge_2 = (UID_2, UID_1, 'identical_nouns')
                idx_edges[edge_1] = True
                idx_edges[edge_2] = True
                idx_identical_noun_edges[edge_1] = True
                idx_identical_noun_edges[edge_2] = True


# Adds edge from reference to referent.
def add_coreference_edges(idx_nodes, idx_edges, idx_coreference_edges):

    # Make XREF from doc_token_index to UID
    xref_doc_token_index_to_UID = {}
    for node_id in idx_nodes:
        graph_node = idx_nodes[node_id]
        doc_token_index = graph_node['doc_token_index']
        xref_doc_token_index_to_UID[doc_token_index] = node_id

    for node_id in idx_nodes:
        graph_node = idx_nodes[node_id]
        all_coref_res_doc_token_index = graph_node['all_coref_res_doc_token_index']
        if all_coref_res_doc_token_index is not False:
            for coref_res_doc_token_index in all_coref_res_doc_token_index:
                coref_res_UID = xref_doc_token_index_to_UID[coref_res_doc_token_index] \
                                    if coref_res_doc_token_index in xref_doc_token_index_to_UID \
                                    else False
                if coref_res_UID:

                    UID_1 = coref_res_UID
                    UID_2 = node_id
                    edge = (UID_1, UID_2, 'coref_res')
                    idx_edges[edge] = True
                    idx_coreference_edges[edge] = True

                    UID_1 = node_id
                    UID_2 = coref_res_UID
                    edge = (UID_1, UID_2, 'coref_res')
                    idx_edges[edge] = True
                    idx_coreference_edges[edge] = True

def render_graph_graphviz(graph_num, idx_nodes, idx_edges, idx_word_UIDs, idx_main_UIDs, idx_copula_edges, idx_target_edges,
                          idx_orig_order_edges,  idx_wh_word_edges, idx_identical_noun_edges, idx_coreference_edges):

    idx_colors = {
        'default':('#DDDDDD', '#DDDDDD', 'white', 'ellipse'),
        'word_UID':('black', 'black', 'white', 'ellipse'),
        'main_UID':('#EFEFEF', 'black', '#EFEFEF', 'circle'),
        'root':('red', 'black', 'aliceblue', 'tripleoctagon'),
        'main_noun':('red', 'black', 'honeydew', 'box'),
        'main_verb':('red', 'black', 'aliceblue', 'octagon'),
        'main_adjective':('red', 'black', 'thistle', 'egg'),
        'subject':('green', 'black', 'palegreen', 'box'),
        'head_different':('red', 'black', '#DDDDDD', 'doubleoctagon'),
        'noun':('palegreen', 'black', 'honeydew', 'box'),
        'verb':('lightblue', 'black', 'aliceblue', 'octagon'),
        'adjective':('violet', 'black', 'thistle', 'egg'),
    }


    dot = graphviz.Digraph('sentence graph ' + str(graph_num), comment='')
    dot.format = 'png'
    dot.engine = 'dot'  # "dot" seems to work the best.

    # Add node-info
    for node_id in idx_nodes:
        color = idx_colors['default'][0]
        fontcolor = idx_colors['default'][1]
        fillcolor = idx_colors['default'][2]
        shape = idx_colors['default'][3]
        if node_id in idx_word_UIDs:
            color = idx_colors['word_UID'][0]
            fontcolor = idx_colors['word_UID'][1]
            fillcolor = idx_colors['word_UID'][2]
            shape = idx_colors['word_UID'][3]
        if node_id in idx_main_UIDs:
            color = idx_colors['main_UID'][0]
            fontcolor = idx_colors['main_UID'][1]
            fillcolor = idx_colors['main_UID'][2]
            shape = idx_colors['main_UID'][3]
            if idx_main_UIDs[node_id] == 'root':
                color = idx_colors['root'][0]
                fontcolor = idx_colors['root'][1]
                fillcolor = idx_colors['root'][2]
                shape = idx_colors['root'][3]
            elif idx_main_UIDs[node_id] == 'main_noun':
                color = idx_colors['main_noun'][0]
                fontcolor = idx_colors['main_noun'][1]
                fillcolor = idx_colors['main_noun'][2]
                shape = idx_colors['main_noun'][3]
            elif idx_main_UIDs[node_id] == 'main_verb':
                color = idx_colors['main_verb'][0]
                fontcolor = idx_colors['main_verb'][1]
                fillcolor = idx_colors['main_verb'][2]
                shape = idx_colors['main_verb'][3]
            elif idx_main_UIDs[node_id] == 'main_adjective':
                color = idx_colors['main_adjective'][0]
                fontcolor = idx_colors['main_adjective'][1]
                fillcolor = idx_colors['main_adjective'][2]
                shape = idx_colors['main_adjective'][3]
            elif idx_main_UIDs[node_id] == 'subject':
                color = idx_colors['subject'][0]
                fontcolor = idx_colors['subject'][1]
                fillcolor = idx_colors['subject'][2]
                shape = idx_colors['subject'][3]
            elif idx_main_UIDs[node_id] == 'head_different_subsentence':
                color = idx_colors['head_different'][0]
                fontcolor = idx_colors['head_different'][1]
                fillcolor = idx_colors['head_different'][2]
                shape = idx_colors['head_different'][3]
            elif idx_main_UIDs[node_id] == 'noun':
                color = idx_colors['noun'][0]
                fontcolor = idx_colors['noun'][1]
                fillcolor = idx_colors['noun'][2]
                shape = idx_colors['noun'][3]
            elif idx_main_UIDs[node_id] == 'verb':
                color = idx_colors['verb'][0]
                fontcolor = idx_colors['verb'][1]
                fillcolor = idx_colors['verb'][2]
                shape = idx_colors['verb'][3]
            elif idx_main_UIDs[node_id] == 'adjective':
                color = idx_colors['adjective'][0]
                fontcolor = idx_colors['adjective'][1]
                fillcolor = idx_colors['adjective'][2]
                shape = idx_colors['adjective'][3]

        dot.node(node_id, fontcolor=fontcolor, color=color, fillcolor=fillcolor, style='filled', shape=shape,
                 label=idx_nodes[node_id]['graph_label'])  # + ' ' + node_id)

    # Add WH-resolution edges.
    for edge in idx_wh_word_edges.keys():
        label = edge[2]
        dot.edge(edge[0], edge[1], color='yellowgreen', label=label, fontcolor='yellowgreen')

    # Add regular edges.
    for edge in idx_edges.keys():
        color = '#DDDDDD'
        label = ''
        if edge in idx_copula_edges:
            color = 'yellowgreen'
            label = 'copula'
        elif edge in idx_target_edges:
            color = 'crimson'
            label = edge[2]
            if label == 'CONJ':
                color = 'darkgreen'
        elif edge in idx_orig_order_edges:
            color = 'blue'
        elif edge in idx_identical_noun_edges:
            color = 'purple'
        elif edge in idx_coreference_edges:
            color = 'magenta'
        dot.edge(edge[0], edge[1], color=color, label=label, fontcolor=color)

    # print(dot.source)

    dot.render(directory='/tmp/graphviz', view=True)


def render_graph_graphviz_app(file_name_str, idx_nodes, idx_edges, idx_word_UIDs, idx_main_UIDs, idx_copula_edges, idx_target_edges,
                          idx_orig_order_edges, idx_wh_word_edges, idx_identical_noun_edges, idx_coreference_edges):


    idx_colors = {
        'default':('#DDDDDD', '#DDDDDD', 'white', 'ellipse'),
        'word_UID':('black', 'black', 'white', 'ellipse'),
        'main_UID':('#EFEFEF', 'black', '#EFEFEF', 'circle'),
        'root':('red', 'black', 'aliceblue', 'tripleoctagon'),
        'main_noun':('red', 'black', 'honeydew', 'box'),
        'main_verb':('red', 'black', 'aliceblue', 'octagon'),
        'main_adjective':('red', 'black', 'thistle', 'egg'),
        'subject':('green', 'black', 'palegreen', 'box'),
        'head_different':('red', 'black', '#DDDDDD', 'doubleoctagon'),
        'noun':('palegreen', 'black', 'honeydew', 'box'),
        'verb':('lightblue', 'black', 'aliceblue', 'octagon'),
        'adjective':('violet', 'black', 'thistle', 'egg'),
    }


    dot = graphviz.Digraph('sentence graph', comment='')
    dot.engine = 'dot'  # "dot" seems to work the best.

    # Add node-info
    for node_id in idx_nodes:
        color = idx_colors['default'][0]
        fontcolor = idx_colors['default'][1]
        fillcolor = idx_colors['default'][2]
        shape = idx_colors['default'][3]
        if node_id in idx_word_UIDs:
            color = idx_colors['word_UID'][0]
            fontcolor = idx_colors['word_UID'][1]
            fillcolor = idx_colors['word_UID'][2]
            shape = idx_colors['word_UID'][3]
        if node_id in idx_main_UIDs:
            color = idx_colors['main_UID'][0]
            fontcolor = idx_colors['main_UID'][1]
            fillcolor = idx_colors['main_UID'][2]
            shape = idx_colors['main_UID'][3]
            if idx_main_UIDs[node_id] == 'root':
                color = idx_colors['root'][0]
                fontcolor = idx_colors['root'][1]
                fillcolor = idx_colors['root'][2]
                shape = idx_colors['root'][3]
            elif idx_main_UIDs[node_id] == 'main_noun':
                color = idx_colors['main_noun'][0]
                fontcolor = idx_colors['main_noun'][1]
                fillcolor = idx_colors['main_noun'][2]
                shape = idx_colors['main_noun'][3]
            elif idx_main_UIDs[node_id] == 'main_verb':
                color = idx_colors['main_verb'][0]
                fontcolor = idx_colors['main_verb'][1]
                fillcolor = idx_colors['main_verb'][2]
                shape = idx_colors['main_verb'][3]
            elif idx_main_UIDs[node_id] == 'main_adjective':
                color = idx_colors['main_adjective'][0]
                fontcolor = idx_colors['main_adjective'][1]
                fillcolor = idx_colors['main_adjective'][2]
                shape = idx_colors['main_adjective'][3]
            elif idx_main_UIDs[node_id] == 'subject':
                color = idx_colors['subject'][0]
                fontcolor = idx_colors['subject'][1]
                fillcolor = idx_colors['subject'][2]
                shape = idx_colors['subject'][3]
            elif idx_main_UIDs[node_id] == 'head_different_subsentence':
                color = idx_colors['head_different'][0]
                fontcolor = idx_colors['head_different'][1]
                fillcolor = idx_colors['head_different'][2]
                shape = idx_colors['head_different'][3]
            elif idx_main_UIDs[node_id] == 'noun':
                color = idx_colors['noun'][0]
                fontcolor = idx_colors['noun'][1]
                fillcolor = idx_colors['noun'][2]
                shape = idx_colors['noun'][3]
            elif idx_main_UIDs[node_id] == 'verb':
                color = idx_colors['verb'][0]
                fontcolor = idx_colors['verb'][1]
                fillcolor = idx_colors['verb'][2]
                shape = idx_colors['verb'][3]
            elif idx_main_UIDs[node_id] == 'adjective':
                color = idx_colors['adjective'][0]
                fontcolor = idx_colors['adjective'][1]
                fillcolor = idx_colors['adjective'][2]
                shape = idx_colors['adjective'][3]

        dot.node(node_id, fontcolor=fontcolor, color=color, fillcolor=fillcolor, style='filled', shape=shape,
                 label=idx_nodes[node_id]['graph_label'])  # + ' ' + node_id)

    # Add WH-resolution edges.
    for edge in idx_wh_word_edges.keys():
        label = edge[2]
        dot.edge(edge[0], edge[1], color='yellowgreen', label=label, fontcolor='yellowgreen')

    # Add regular edges.
    for edge in idx_edges.keys():
        color = '#DDDDDD'
        label = ''
        if edge in idx_copula_edges:
            color = 'yellowgreen'
            label = 'copula'
        elif edge in idx_target_edges:
            color = 'crimson'
            label = edge[2]
            if label == 'CONJ':
                color = 'darkgreen'
        elif edge in idx_orig_order_edges:
            color = 'blue'
        elif edge in idx_identical_noun_edges:
            color = 'purple'
        elif edge in idx_coreference_edges:
            color = 'magenta'
        dot.edge(edge[0], edge[1], color=color, label=label, fontcolor=color)

    dot.render(outfile=file_name_str, view=False)

    os.unlink(pathlib.PurePath(file_name_str).with_suffix('.gv'))