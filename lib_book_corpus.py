import sys
import os
import io
import pathlib
import subprocess
import re
import csv
import unicodedata
import pickle
import tempfile
import nltk


###################################################################################

# CONFIG

# Directories for the corpus

original_PDF_books_dir = '/home/bryan/Documents/books/ML-DS PDF books/'
dumped_PDF_books_dir = '/home/bryan/Documents/books/ML-DS pdfminer-dump/'




###################################################################################


# The result is ordered by book name
def get_book_name_list_for_corpus():
    book_name_str_recs = []
    with os.scandir(original_PDF_books_dir) as it:
        for entry in it:
            if entry.is_file():
                book_name_str_recs.append(entry.name)
    book_name_str_recs = sorted(book_name_str_recs)

    return book_name_str_recs



####################################################################################

# See https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
def unicodedata_normalize(text):
   return unicodedata.normalize("NFKD", text)


def normalize_text_block(text):

    # Strip whitespace from beginning and end of block.
    text = text.strip()

    # Un-break words hyphenated because of line breaks.
    # THE ARE ALL DIFFERENT CHARACTERS, EVEN THOUGH THEY LOOK THE SAME.
    HYPHEN = '-'
    EN_DASH = '–'
    EM_DASH = '—'
    MINUS_SIGN = '−'
    text = text.replace(HYPHEN + '\n', '').replace(EN_DASH + '\n', '').replace(EM_DASH + '\n', '').replace(MINUS_SIGN + '\n', '')

    text = text.replace('/' + '\n', '').replace('+' + '\n', '')

    # Remove quotes
    text = text.replace('“', ' ').replace('”', ' ').replace('"', ' ')

    # Strip accent marks ("Naïve Bayes" --> "Naive Bayes")
    text = unicodedata_normalize(text)

    # Whitespaces stuff ===================================

    # Try to standardize line endings that aren't sentence endings.
    text = re.sub('([^\.\?\! \n]+) *\n+ *([a-z])', '\\1 \\2', text)

    # Try to standardize sentence endings at line endings.
    text = re.sub('([\.\?\!]) *\n+ *([A-Z0-9])', '\\1 \\2', text)


    # Multiple newlines.
    text = re.sub('\n+', '\n', text)

    # Multiple spaces.
    text = re.sub(' +', ' ', text)

    # Once again, strip whitespace from beginning and end of block.
    text = text.strip()

    return text

def normalize_text_block_list(text_block_list):
    return [normalize_text_block(text_block) for text_block in text_block_list]


# (page_num, font, text)
#       (221, '13.6 UPWMBK+LinLibertine', 'with a class attribute that serves as a reference for the future\n')
def get_text_block_from_text_element_list(text_element_list):
    text_block = ''.join(rec[2] for rec in text_element_list)
    return text_block


# (page_num, font, text)
#       (221, '13.6 UPWMBK+LinLibertine', 'with a class attribute that serves as a reference for the future\n')
def group_text_element_list_by_font(text_element_list):
    font_groups = {}
    for rec in text_element_list:
        font = rec[1]
        if not font in font_groups:
            font_groups[font] = []
        font_groups[font].append(rec)

    return font_groups


# (page_num, font, text)
#       (221, '13.6 UPWMBK+LinLibertine', 'with a class attribute that serves as a reference for the future\n')
def group_text_element_list_by_page(text_element_list):
    page_groups = {}
    for rec in text_element_list:
        page = rec[0]
        if not page in page_groups:
            page_groups[page] = []
        page_groups[page].append(rec)

    return page_groups

# (page_num, font, text)
#       (221, '13.6 UPWMBK+LinLibertine', 'with a class attribute that serves as a reference for the future\n')
#
# We're going to say a "paragraph" is a block of text which ends with a line whose final character is '.', '!', or '?'.
def group_text_element_list_by_paragraph(text_element_list):
    paragraph_groups = {}

    par_num = 0
    cur_par_group = []
    for rec in text_element_list:
        text = rec[2].strip()

        cur_par_group.append(rec)

        if text.endswith('.') or text.endswith('!') or text.endswith('?'):
            paragraph_groups[par_num] = cur_par_group
            par_num += 1
            cur_par_group = []

    # Cleanup the (likely) straggler.
    paragraph_groups[par_num] = cur_par_group

    return paragraph_groups

def group_text_element_list_by_font_then_page(text_element_list):
    font_groups = group_text_element_list_by_font(text_element_list)
    for font in font_groups:
        page_groups = group_text_element_list_by_page(font_groups[font])
        font_groups[font] = page_groups

    return font_groups

def group_text_element_list_by_font_then_paragraph(text_element_list):
    font_groups = group_text_element_list_by_font(text_element_list)
    for font in font_groups:
        paragraph_groups = group_text_element_list_by_paragraph(font_groups[font])
        font_groups[font] = paragraph_groups

    return font_groups


def chunk_text_blocks_until_length(text_block_list, min_length=5000):
    new_text_block_list = []

    cur_new_block = False
    for text_block in text_block_list:
        if cur_new_block is False:
            cur_new_block = text_block
        else:
            cur_new_block = cur_new_block + '\n' + text_block
        if len(cur_new_block) > min_length:
            new_text_block_list.append(cur_new_block)
            cur_new_block = False
    # Cleanup
    if cur_new_block:
        new_text_block_list.append(cur_new_block)

    return new_text_block_list


def extract_all_normalized_text_block_list(pdf_file_str, min_length=5000):
    fontinfo_recs, book_text_element_list = pickle_load_font_and_text_information_from_PDF_file(pdf_file_str)
    font_and_paragraph_groups = group_text_element_list_by_font_then_paragraph(book_text_element_list)

    normalized_text_block_list = []

    for fontinfo_rec in fontinfo_recs:
        font = fontinfo_rec[0]
        text_block_list = list(get_text_block_from_text_element_list(font_and_paragraph_groups[font][par_num]) for par_num in font_and_paragraph_groups[font])
        text_block_list = chunk_text_blocks_until_length(text_block_list, min_length=min_length)
        text_block_list = normalize_text_block_list(text_block_list)
        for text_block in text_block_list:
            normalized_text_block_list.append(text_block)

    return normalized_text_block_list

# Check first to see if have already extracted text and if so use it.
def fast_get_text_from_PDF(pdf_file_str, save_results=True, look_for_already_processed=True):
    pdf_text_dump_path = os.path.join(dumped_PDF_books_dir, pdf_file_str + '.text_dump.txt')

    if look_for_already_processed:
        if os.path.isfile(pdf_text_dump_path):
            text = open(pdf_text_dump_path, "r").read()
            return text

    pdf_path = os.path.join(original_PDF_books_dir, pdf_file_str)
    temp_file_wrapper = tempfile.NamedTemporaryFile(suffix='.txt', prefix='fast_get_text_from_PDF_')
    file_name_str = temp_file_wrapper.name
    shell_cmd = 'pdftotext "' + pdf_path + '" ' + file_name_str
    os.system(shell_cmd)

    text = open(file_name_str, "r").read()

    # Replace form feed with space.
    text = re.sub('\f', ' ', text)

    text = unicodedata_normalize(text)

    if save_results:
        open(pdf_text_dump_path, "w").write(text)

    return text

def sanitize_sentence(sentence):
    # Remove everything except words, and convert to lowercase.
    sentence = re.sub('[^a-zA-z0-9]', ' ', sentence).lower()
    # Replace double spaces with singles.
    sentence = re.sub(' +', ' ', sentence)

    return sentence


# Also provides the SpaCy results for the sentences.
# Throws out interrogative and imperatives.
# Try to break up chunks in nice ways by looking for period followed by whitespace.
def extract_valid_declarative_sentences_from_text(spacy_client, text, text_block_char_length=10000):
    double_newline_splits_list = text.split('\n\n')
    text_block_list = []
    for text in double_newline_splits_list:
        while len(text) > text_block_char_length:
            text_block_list.append(text[:text_block_char_length])
            text = text[text_block_char_length:]
        # Cleanup
        if len(text) > 0:
            text_block_list.append(text)

    weird_token_tags = ['SP', '_SP', 'LS', '$(', '$,', 'SYM', 'X', 'EOL', 'SPACE', ':', 'HYPH', 'NIL', 'ADD', 'NFP', 'XX', '$.', 'ITJ']
    weird_token_deprels = ['punct', 'dep', 'advmod', 'intj', 'list', 'meta', 'orphan']
    formula_tags = ['LS', '$(', '$,', 'SYM', 'X', ':', 'HYPH']
    formula_deprels = ['dep', 'list', 'meta', 'orphan']

    valid_sentences_list = []
    valid_count = 1
    invalid_count = 1

    text_block_counter = 0
    num_text_blocks = len(text_block_list)
    for text_block in text_block_list:
        text_block_counter += 1
        print("Processing text_block #", text_block_counter, "of", num_text_blocks)

        doc = spacy_client(text_block)
        for sent in doc.sents:
            # See if we have a real sentence.
            tokens = list(token for token in sent)
            meaningful_tokens = list(token for token in sent \
                    if (token.tag_ not in weird_token_tags and token.dep_ not in weird_token_deprels))
            formula_token_count = sum(1 for token in sent if (token.tag_ in formula_tags or token.dep_ in formula_deprels))
            period_count = sum(1 for token in sent if token.text == '.')
            idx_deprels = dict((token.dep_.lower(), token.tag_) for token in sent)
            verb_POS_list = ['MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

            bool_is_valid_sentence = True
            root_deprel = 'root' if 'root' in idx_deprels else False
            if period_count > 3 or formula_token_count > 4:
                bool_is_valid_sentence = False
                # print()
                # print("================================================")
                # print(sent)
            elif root_deprel is False or idx_deprels[root_deprel] not in verb_POS_list:
                bool_is_valid_sentence = False
            elif tokens[-1].text == '?':
                bool_is_valid_sentence = False
            elif meaningful_tokens[0].tag_ in verb_POS_list:
                bool_is_valid_sentence = False

            # print("==================================================")
            # print(sent)
            # print(bool_is_valid_sentence)
            # print(list((token.text, token.tag_, token.pos_, token.dep_) for token in meaningful_tokens))

            if bool_is_valid_sentence:
                valid_count += 1
                valid_sentences_list.append(sent.text.strip().replace("\n", " "))
            else:
                invalid_count += 1

        #print(valid_count/(valid_count + invalid_count))

    return valid_sentences_list


def pickle_dump_valid_sentences_text_list(pdf_file_str, valid_sentences_text_list):
    pickled_dump_path = os.path.join(dumped_PDF_books_dir, pdf_file_str + '.valid_sentences_text_list.pickled')
    pickle.dump(valid_sentences_text_list, open(pickled_dump_path, 'wb'))


def pickle_load_valid_sentences_text_list(pdf_file_str):
    pickled_dump_str = os.path.join(dumped_PDF_books_dir, pdf_file_str + '.valid_sentences_text_list.pickled')
    valid_sentences_text_list = pickle.load(open(pickled_dump_str, 'rb'))
    return valid_sentences_text_list


def make_text_from_valid_sentences(valid_sentences_text_list):
    return '\n'.join(valid_sentences_text_list)


# Fast, but not particularly accurate.
def fast_split_text_into_sentences(text):
    return list(nltk.tokenize.sent_tokenize(text))


# Counts POS by word.
def fast_get_POS_counts_by_word(text):
    idx_POS_counts_by_word = {}

    noun_list = ['WP', 'NN', 'NNS', 'NNP', 'NNPS', 'PRP']
    adj_list = ['JJ', 'JJR', 'JJS']
    verb_list = ['MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    tokens = nltk.pos_tag(nltk.tokenize.word_tokenize(text))
    for token in tokens:
        word = token[0].lower()
        POS_tag = token[1]

        POS_cat = 'OTHER'
        if POS_tag in noun_list:
            POS_cat = 'NOUN'
        elif POS_tag in adj_list:
            POS_cat = 'ADJ'
        elif POS_tag in verb_list:
            POS_cat = 'VERB'

        if not word in idx_POS_counts_by_word:
            idx_POS_counts_by_word[word] = {}
        if not POS_cat in idx_POS_counts_by_word[word]:
            idx_POS_counts_by_word[word][POS_cat] = 0
        idx_POS_counts_by_word[word][POS_cat] += 1

    # Sort by most common.
    for word in idx_POS_counts_by_word:
        POS_recs = list((POS_cat, idx_POS_counts_by_word[word][POS_cat]) for POS_cat in idx_POS_counts_by_word[word])
        POS_recs = sorted(POS_recs, key=lambda rec: rec[1], reverse=True)
        idx_POS_counts_by_word[word] = POS_recs

    return idx_POS_counts_by_word


def get_highest_scoring_disjoint_terms(term_list, idx_candidate_term_scores):
    term_graph = {}
    # First pass.
    for term in term_list:
        term_graph[term] = {'term': term, 'score': idx_candidate_term_scores[term], 'words':set(term.split())}


    # Greedy method to find high-scoring and disjoint terms.
    sorted_term_node_list = list(term_graph[term] for term in term_graph)
    sorted_term_node_list = sorted(sorted_term_node_list, key= lambda rec: rec['score'])

    best_disjoint_term_nodes = []
    words_used = set()
    while len(sorted_term_node_list) > 0:
        best_term_node = sorted_term_node_list.pop()
        words_used = words_used | best_term_node['words']
        best_disjoint_term_nodes.append(best_term_node)
        # Remove all other terms which share a word with this one.
        sorted_term_node_list = list(term_graph[term] for term in term_graph if (words_used & term_graph[term]['words']) == set() )
        sorted_term_node_list = sorted(sorted_term_node_list, key= lambda rec: rec['score'])

    highest_scoring_disjoint_terms = []
    total_score = 0
    for term_node in best_disjoint_term_nodes:
        highest_scoring_disjoint_terms.append(term_node['term'])
        total_score += term_node['score']

    return highest_scoring_disjoint_terms, total_score


def normalize_term(term):
    term = ' '.join(word for word in re.split('[^a-zA-Z]', term) if word.strip() != '').lower()

    return term


# "Kind of" recursive.
def are_synonym_singular_plural(term_1, term_2):
    if term_1 == term_2:
        return True

    if term_1 + 's' == term_2:
        return True
    elif term_2 + 's' == term_1:
        return True
    elif term_1.endswith('y') and term_1[:len(term_1) - 1] + 'ies' == term_2:
        return True
    elif term_2.endswith('y') and term_2[:len(term_2) - 1] + 'ies' == term_1:
        return True
    else:
        # Split terms and compare each word, again using are_synonym_singular_plural().
        words_1 = term_1.split()
        words_2 = term_2.split()
        bool_good = False
        if len(words_1) > 1 and len(words_1) == len(words_2):
            bool_good = True
            for ii in range(len(words_1)):
                bool_good = bool_good and are_synonym_singular_plural(words_1[ii], words_2[ii])
                if not bool_good:
                    break
        return bool_good


def make_synonym_partition_groups(idx_synonyms, idx_scores_by_term):
    synonym_groups = []
    for term in idx_synonyms:
        synonym_groups.append(set([term]) | set(idx_synonyms[term].keys()))

    partition_groups = []
    idx_jjs_unioned = {}
    for ii in range(len(synonym_groups)):
        if ii in idx_jjs_unioned:
            continue
        group_1 = synonym_groups[ii]
        for jj in range(ii + 1, len(synonym_groups)):
            group_2 = synonym_groups[jj]
            if len(group_1 & group_2) > 0:
                group_1 = group_1 | group_2
                idx_jjs_unioned[jj] = True
        partition_groups.append(group_1)

    # Make a dictionary of partition groups indexed by the highest-scoring term.
    # Also make dictionary of terms giving the highest-scoring of the partition group they're in.
    idx_synonym_partition_groups = {}
    idx_synonym_to_partition_group_id = {}
    for group in partition_groups:
        group_terms = list((term, len(term), idx_scores_by_term[term]) for term in group)
        #group_id = sorted(group_terms, key=lambda rec: rec[1], reverse=True)[0][0]         # VERSION #1: longest member
        group_id = sorted(group_terms, key=lambda rec: rec[2], reverse=True)[0][0]          # VERSION #2: highest-scoring
        idx_synonym_partition_groups[group_id] = group
        for term in group:
            idx_synonym_to_partition_group_id[term] = group_id

    return idx_synonym_partition_groups, idx_synonym_to_partition_group_id

def get_jaccard_distances_between_terms(idx_term_candidate_context_terms, term_1, term_2):
    context_set_1 = idx_term_candidate_context_terms[term_1] | set() # Prevents mutating set.
    context_set_2 = idx_term_candidate_context_terms[term_2] | set() # Prevents mutating set.

    if context_set_1 == set():
        context_set_1 = set(['__DUMMY_1__'])
    if context_set_2 == set():
        context_set_2 = set(['__DUMMY_2__'])

    # NOTE: The "directed distance" seems to give junky results.

    J_sim = len(context_set_1 & context_set_2)/len(context_set_1 | context_set_2)

    # Get distance from similarity.
    J_dist = 1 - J_sim

    return J_dist

# edges = [edge_1, edge_2, ...]
# Each resulting component is a set of nodes.
def partition_nodes_into_connected_components(edges):
    idx_nodes = {}
    idx_edges_by_node = {}
    for edge in edges:
        term_1 = edge[0]
        term_2 = edge[1]
        idx_nodes[term_1] = True
        idx_nodes[term_2] = True
        if not term_1 in idx_edges_by_node:
            idx_edges_by_node[term_1] = []
        if not term_2 in idx_edges_by_node:
            idx_edges_by_node[term_2] = []
        idx_edges_by_node[term_1].append(term_2)
        idx_edges_by_node[term_2].append(term_1)

    components = []

    idx_seen_nodes = {}
    while True > 0:
        seed_node = False
        for node in idx_nodes:
            if not node in idx_seen_nodes:
                seed_node = node
                break
        if seed_node is False:
            break

        node_queue = [seed_node]
        idx_seen_nodes[seed_node] = True
        cur_component = set()
        while len(node_queue) > 0:
            node = node_queue.pop()
            # Add node to current component.
            cur_component.add(node)
            # Add edge targets to queue if we haven't already seen it.
            for target_node in idx_edges_by_node[node]:
                if target_node not in idx_seen_nodes:
                    node_queue.append(target_node)
                    idx_seen_nodes[target_node] = True
        components.append(cur_component)

    return components


###############################################################################################################


# # TEST

# text = """
# 8. Tree-Based Methods

# (f) Apply the cv.tree() function to the training set in order to
# determine the optimal tree size.
# (g) Produce a plot with tree size on the x-axis and cross-validated
# classification error rate on the y-axis.
# (h) Which tree size corresponds to the lowest cross-validated classification error rate?
# (i) Produce a pruned tree corresponding to the optimal tree size
# obtained using cross-validation. If cross-validation does not lead
# to selection of a pruned tree, then create a pruned tree with five
# terminal nodes.
# (j) Compare the training error rates between the pruned and unpruned trees. Which is higher?
# (k) Compare the test error rates between the pruned and unpruned
# trees. Which is higher?
# 10. We now use boosting to predict Salary in the Hitters data set.
# (a) Remove the observations for whom the salary information is
# unknown, and then log-transform the salaries.
# (b) Create a training set consisting of the first 200 observations, and
# a test set consisting of the remaining observations.
# (c) Perform boosting on the training set with 1,000 trees for a range
# of values of the shrinkage parameter λ. Produce a plot with
# different shrinkage values on the x-axis and the corresponding
# training set MSE on the y-axis.
# (d) Produce a plot with different shrinkage values on the x-axis and
# the corresponding test set MSE on the y-axis.
# (e) Compare the test MSE of boosting to the test MSE that results
# from applying two of the regression approaches seen in
# Chapters 3 and 6.
# (f) Which variables appear to be the most important predictors in
# the boosted model?
# (g) Now apply bagging to the training set. What is the test set MSE
# for this approach?
# 11. This question uses the Caravan data set.
# (a) Create a training set consisting of the first 1,000 observations,
# and a test set consisting of the remaining observations.
# (b) Fit a boosting model to the training set with Purchase as the
# response and the other variables as predictors. Use 1,000 trees,
# and a shrinkage value of 0.01. Which predictors appear to be
# the most important?

#  8.4 Exercises

# 365

# (c) Use the boosting model to predict the response on the test data.
# Predict that a person will make a purchase if the estimated probability of purchase is greater than 20 %. Form a confusion matrix. What fraction of the people predicted to make a purchase
# do in fact make one? How does this compare with the results
# obtained from applying KNN or logistic regression to this data
# set?
# 12. Apply boosting, bagging, random forests, and BART to a data set
# of your choice. Be sure to fit the models on a training set and to
# evaluate their performance on a test set. How accurate are the results
# compared to simple methods like linear or logistic regression? Which
# of these approaches yields the best performance?

#  9
# Support Vector Machines

# In this chapter, we discuss the support vector machine (SVM), an approach
# for classification that was developed in the computer science community in
# the 1990s and that has grown in popularity since then. SVMs have been
# shown to perform well in a variety of settings, and are often considered one
# of the best “out of the box” classifiers.
# The support vector machine is a generalization of a simple and intuitive classifier called the maximal margin classifier , which we introduce in
# Section 9.1. Though it is elegant and simple, we will see that this classifier
# unfortunately cannot be applied to most data sets, since it requires that
# the classes be separable by a linear boundary. In Section 9.2, we introduce
# the support vector classifier , an extension of the maximal margin classifier
# that can be applied in a broader range of cases. Section 9.3 introduces the
# support vector machine, which is a further extension of the support vector classifier in order to accommodate non-linear class boundaries. Support
# vector machines are intended for the binary classification setting in which
# there are two classes; in Section 9.4 we discuss extensions of support vector
# machines to the case of more than two classes. In Section 9.5 we discuss
# the close connections between support vector machines and other statistical
# methods such as logistic regression.
# People often loosely refer to the maximal margin classifier, the support
# vector classifier, and the support vector machine as “support vector
# machines”. To avoid confusion, we will carefully distinguish between these
# three notions in this chapter.

# © Springer Science+Business Media, LLC, part of Springer Nature 2021
# G. James et al., An Introduction to Statistical Learning, Springer Texts in Statistics,
# https://doi.org/10.1007/978-1-0716-1418-1_9

# 367

#  368

# 9. Support Vector Machines
# """

# import spacy
# spacy_client = spacy.load("en_core_web_sm")
# res = extract_valid_declarative_sentences_from_text(spacy_client, text, text_block_char_length=100000)

# for rec in res:
#     print("======================================================================")
#     print(rec)
