from lib_vector_clustering import *
from lib_embeddings import *
from lib_disk_cache import *
from lib_book_corpus import *
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
import numpy as np
import math
import graphviz
import os
import spacy
import statistics
import sklearn.metrics as sklearn_metrics
from sklearn.linear_model import LinearRegression
from matplotlib import cm

####################################################################################################
### CONFIG #########################################################################################
####################################################################################################

os.environ["TOKENIZERS_PARALLELISM"] = "false"

cache_path = '/tmp/topic_model_cache'

### END CONFIG #####################################################################################
####################################################################################################


####################################################################################################
####################################################################################################
####################################################################################################


def get_lookforward_sentence_window_sums(vector_list, window_size, last_window_size_sums):
    delta = np.zeros((vector_list.shape[0], vector_list.shape[1]))
    delta[:vector_list.shape[0] - window_size + 1] = vector_list[window_size - 1:]

    if window_size == 1:
        return delta
    else:
        return last_window_size_sums + delta


def get_lookback_sentence_window_sums(vector_list, window_size, last_window_size_sums):
    delta = np.zeros((vector_list.shape[0], vector_list.shape[1]))
    delta[window_size:] = vector_list[:vector_list.shape[0] - window_size]

    if window_size == 1:
        return delta
    else:
        return last_window_size_sums + delta


def get_extrema(values_list):
    arg_max = np.argmax(values_list)
    arg_min = np.argmin(values_list)
    v_max = values_list[arg_max]
    v_min = values_list[arg_min]

    return ((arg_max, v_max), (arg_min, v_min))


def find_extrema_in_sliding_windows(values_list, window_size):
    window_size = int(window_size)

    extrema_set = set()
    for ii in range(len(values_list) - window_size):
        window_start = ii
        window_end = window_start + window_size - 1
        window_vals = values_list[window_start:window_end + 1]
        window_extrema = get_extrema(window_vals)

        # Only consider ones where the max is near the center of the window.
        arg_max = window_extrema[0][0]
        if not abs(arg_max - 0.5*window_size) < 1:
            continue

        # Convert indices to original values_list indices.
        ind_of_max = window_start + window_extrema[0][0]
        v_max = window_extrema[0][1]
        ind_of_min = window_start + window_extrema[1][0]
        v_min = window_extrema[1][1]

        # Only consider ones where v_max is greater than mean.
        mean_value = np.mean(values_list)
        if v_max < mean_value:
            continue

        extrema = ((ind_of_max, v_max), (ind_of_min, v_min))
        extrema_set.add(extrema)

    extrema_list = list(extrema_set)
    extrema_list = sorted(extrema_list, key=lambda rec:(min(rec[0][0], rec[1][0])))

    return extrema_list

def extract_useful_extrema_in_sliding_windows(distances, window_size):
    extrema_list = find_extrema_in_sliding_windows(distances, window_size)

    extrema_maxima_set = set(rec[0][0] for rec in extrema_list) # Remove duplicates this way

    extrema_maxima_list = list(extrema_maxima_set)
    extrema_maxima_list.sort()

    return extrema_maxima_list


def get_topic_change_extrema_presence_grid(embedded_sentences, min_window_size):
    num_sentences = len(embedded_sentences)

    # Windows are decreasing in size as the number of iterations increases.
    window_size_list = list(range(min_window_size, int(num_sentences/5)))

    # Make sure is sorted from smallest to largest.
    window_size_list.sort()

    lookforward_sums = False
    lookback_sums = False
    # Initialize lookforward_sums and lookback_sums for windows smaller than the smallest one we're using...
    for ii in range(1, min_window_size):
        lookforward_sums = get_lookforward_sentence_window_sums(embedded_sentences, ii, lookforward_sums)
        lookback_sums = get_lookback_sentence_window_sums(embedded_sentences, ii, lookback_sums)

    extrema_presence_grid = np.zeros((embedded_sentences.shape[0], max(window_size_list) + 1))

    res_by_window_size = {}
    for window_size in window_size_list:
        print("window_size:", window_size)

        lookforward_sums = get_lookforward_sentence_window_sums(embedded_sentences, window_size, lookforward_sums)
        lookback_sums = get_lookback_sentence_window_sums(embedded_sentences, window_size, lookback_sums)

        # Always take the "mean" by dividing by window size.
        lookahead_window_means = lookforward_sums/window_size
        lookback_window_means = lookback_sums/window_size

        # Local distances ------------------------------------------------------------------------------

        distances_list = np.array(list(1 - get_vector_similarity(lookahead_window_means[ii], lookback_window_means[ii]) for ii in range(num_sentences)))

        #-----------------------------------------------------------------------------------------------

        # Add extrema to results
        extrema_list = extract_useful_extrema_in_sliding_windows(distances_list, window_size)

        res_by_window_size[window_size] = extrema_list

        for extremum in extrema_list:
            extrema_presence_grid[extremum, window_size] = 1


    return extrema_presence_grid


# The corpus is the collection of child nodes...
def calc_TF_IDFs_for_child_nodes(tree_node):

    tfs_by_doc_list = [child_node['idx_term_ind_scaled_freqs'] for child_node in tree_node['children']]

    num_docs = len(tfs_by_doc_list)

    idx_doc_presence_counts = {}
    for idx_doc_tfs_by_term in tfs_by_doc_list:
        for term_ind in idx_doc_tfs_by_term:
            if not term_ind in idx_doc_presence_counts:
                idx_doc_presence_counts[term_ind] = 0
            idx_doc_presence_counts[term_ind] += 1

    # Standard version of IDF
    # See https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency_2
    idx_idfs_by_term = {term_ind:math.log(num_docs/idx_doc_presence_counts[term_ind]) for term_ind in idx_doc_presence_counts}

    # Calculate tf-idf by term, for each doc in the corpus.
    idx_tf_idfs_by_term_for_docs = []
    for child_ind in range(len(tfs_by_doc_list)):
        idx_doc_tfs_by_term = tfs_by_doc_list[child_ind]
        idx_tf_idfs = {term_ind:idx_doc_tfs_by_term[term_ind]*idx_idfs_by_term[term_ind] for term_ind in idx_doc_tfs_by_term}
        # Set in child
        tree_node['children'][child_ind]['idx_tf_idfs'] = idx_tf_idfs


def print_tree_node(tree_node, sentence_list, terms_list, single_node=False):

    level_spacer = "\t"*tree_node['level']
    if single_node:
        level_spacer = ''

    print()

    # Main info about node
    print(level_spacer, "node_id:" + str(tree_node['node_id']), "parent_node_id:" + str(tree_node['parent_node_id']),
            tree_node['level'], tree_node['start_at'], tree_node['end_at'])
    print(level_spacer, "# of child nodes:", len(tree_node['children']))
    print(level_spacer, '--------------------------------')

    # Best term for node mean sentence
    print(level_spacer, "T-MEAN :", terms_list[tree_node['best_term_ind_for_mean']])
    print(level_spacer, "T-ALT-MEAN :", terms_list[tree_node['best_term_ind_for_alt_mean']])

    print(level_spacer, '--------------------------------')

    # Similarities of children to one another
    num_children = len(tree_node['children'])
    if num_children > 0:
        for child_ind in range(num_children):
            child_node = tree_node['children'][child_ind]
            child_node_frac = (1 + child_node['end_at'] - child_node['start_at'])/(1 + tree_node['end_at'] - tree_node['start_at'])
            child_node_frac = round(child_node_frac, 4)
            similarity_to_parent = get_inner_product_combinations([child_node['mean_vector']], [tree_node['mean_vector']])[0,0]
            similarity_to_prev_child = False
            if child_ind > 0:
                prev_child = tree_node['children'][child_ind - 1]
                similarity_to_prev_child = get_inner_product_combinations([child_node['mean_vector']], [prev_child['mean_vector']])[0,0]
            print(level_spacer, 'Child #' + str(child_ind), 'length frac:', child_node_frac, similarity_to_parent, similarity_to_prev_child)

    print(level_spacer, '--------------------------------')

    num_terms_to_show = 10

    # List of terms and their attributes
    list_of_attributes_by_term = tree_node['list_of_attributes_by_term']

    # Sort
    # DON'T TOUCH THIS WITHOUT TESTING
    list_of_attributes_by_term = sorted(list_of_attributes_by_term, key=lambda rec:rec['sort_rank_sim_to_alt_mean'])

    for attribute_rec in list_of_attributes_by_term[:num_terms_to_show]:
        print(level_spacer,
                    attribute_rec['sort_rank_tf_idf'],
                    attribute_rec['sort_rank_sim_to_alt_mean'],
                    attribute_rec['sort_rank_sim_to_alt_mean_ATE_count'],
                    '/',
                    attribute_rec['term'],
                    '/',
                    'RANK:' + str(attribute_rec['sort_rank_tf_idf']),
                    '\t###\t',
                    "count:", attribute_rec['ATE_count'], "count_aug:", attribute_rec['ATE_count_augmented'],
                    "tf_idf:", attribute_rec['tf_idf_rel_siblings'],
                    "sim_to_alt_mean:", attribute_rec['sim_to_alt_mean'])

    # # Node exemplar sentence
    # print(level_spacer+ '\t EXEMPLAR:', tree_node['exemplar_ind'], '###', sentence_list[tree_node['exemplar_ind']].replace("\n", " "))

    # print()

    # num_sentences_to_show = min(len(tree_node['sentence_inds_sorted_by_sim_alt_mean_vector']), 10)
    # for ii in range(num_sentences_to_show):
    #     sentence_ind = tree_node['sentence_inds_sorted_by_sim_alt_mean_vector'][ii]
    #     print(level_spacer, "Rank #", ii + 1, 'Sent #', sentence_ind, '###', sentence_list[sentence_ind].replace("\n", " "))





def recursive_tree_print(tree_node, sentence_list, terms_list, last_start_at=-1):
    print_tree_node(tree_node, sentence_list, terms_list)

    for child_node in tree_node['children']:
        recursive_tree_print(child_node, sentence_list, terms_list, last_start_at=tree_node['start_at'])


def recursive_tree_debug_print(tree_node):
    print("\t"*tree_node['level'], tree_node['node_id'], tree_node['start_at'], tree_node['end_at'], len(tree_node['idx_ATE_term_counts']))

    for child_node in tree_node['children']:
        recursive_tree_debug_print(child_node)


# Recursive
def get_tree_nodes_by_node_id(tree_node, idx_tree_nodes_by_node_id=False):
    if idx_tree_nodes_by_node_id is False:
        idx_tree_nodes_by_node_id = {}

    idx_tree_nodes_by_node_id[tree_node['node_id']] = tree_node

    # Recurse
    for child_node in tree_node['children']:
        get_tree_nodes_by_node_id(child_node, idx_tree_nodes_by_node_id=idx_tree_nodes_by_node_id)

    return idx_tree_nodes_by_node_id


# min_node_size needs to be at least 1 to avoid errors.
def recursive_build_tree_from_extrema_counts(parent_extrema_counts, parent_node, min_node_size):
    divider_index = np.argmax(parent_extrema_counts)
    # If divider_index == 0, then we are probably in a right split of a previous split,
    #   so disallow this guy and use the next biggest one.
    if divider_index == 0:
        divider_index = 1 + np.argmax(parent_extrema_counts[1:])

    # Parent info
    parent_path_to_top = parent_node['path_to_top']
    parent_level = parent_node['level']
    parent_start_at = parent_node['start_at']
    parent_end_at = parent_node['end_at']

    # Split into two groups.
    # The sentence at divider_index belongs to the second group,
    #   but divider_index will not be included in reduced extrema counts of either group.
    extrema_counts_left = parent_extrema_counts[:divider_index]
    extrema_counts_right = parent_extrema_counts[divider_index:]

    left_node = {
        'level':parent_level + 1,
        'LR':'L',
        'path_to_top':parent_path_to_top + '-L',
        'start_at':parent_start_at,
        'end_at':parent_start_at + divider_index - 1,
        'children':[]
    }

    right_node = {
        'level':parent_level + 1,
        'LR':'R',
        'path_to_top':parent_path_to_top + '-R',
        'start_at':parent_start_at + divider_index,
        'end_at':parent_end_at,
        'children':[]
    }

    #print("\t"*parent_level, parent_start_at, parent_end_at, left_node['start_at'], left_node['end_at'], min_node_size)

    # Add left side to children, and recurse if sufficiently big.
    if 1 + left_node['end_at'] - left_node['start_at'] >= min_node_size:
        parent_node['children'].append(left_node)
        if len(extrema_counts_left) >= 2:
            recursive_build_tree_from_extrema_counts(extrema_counts_left, left_node, min_node_size)

    # Add right side to children, and recurse if sufficiently big.
    if 1 + right_node['end_at'] - right_node['start_at'] >= min_node_size:
        parent_node['children'].append(right_node)
        if len(extrema_counts_right) >= 2:
            recursive_build_tree_from_extrema_counts(extrema_counts_right, right_node, min_node_size)


# This will change the tree for an (at most) binary try to an (at most) n-ary tree.
# A node is useless if it is an only child.
def recursive_remove_useless_children(tree_node, level=0):
    tree_node['level'] = level

    bool_changed = True
    while bool_changed:
        bool_changed = False
        if len(tree_node['children']) == 1:
            bool_changed = True
            tree_node['children'] = tree_node['children'][0]['children']

    # Recurse
    for child_node in tree_node['children']:
        recursive_remove_useless_children(child_node, level=level+1)

    # Sort children by start_at just in case...
    tree_node['children'] = sorted(tree_node['children'], key=lambda child_node:child_node['start_at'])


def recursive_expand_children_to_fill_parent(tree_node):
    if len(tree_node['children']) > 0:
        first_child = tree_node['children'][0]
        last_child = tree_node['children'][len(tree_node['children']) - 1]

        first_child['start_at'] = tree_node['start_at']
        last_child['end_at'] = tree_node['end_at']

    # Recurse
    for child_node in tree_node['children']:
        recursive_expand_children_to_fill_parent(child_node)


def recursive_add_node_ids_and_links_to_parent(tree_node, max_node_id=0):
    max_node_id += 1

    node_id = max_node_id
    tree_node['node_id'] = node_id

    if node_id == 1:
        tree_node['parent_node_id'] = 0
        tree_node['parent'] = False

    for child_node in tree_node['children']:
        child_node['parent_node_id'] = node_id
        child_node['parent'] = tree_node

        # Recurse
        max_node_id = recursive_add_node_ids_and_links_to_parent(child_node, max_node_id=max_node_id)

    return max_node_id


def recursive_add_term_count_rollups(tree_node, idx_term_inds_by_sentence_num, idx_term_ind_to_subterm_inds):
    bool_has_terms = False

    # Initialize from-scrach ATE term counts
    idx_ATE_term_counts = {}
    if len(tree_node['children']) == 0:
        # Get terms which appear in the node.
        for sent_ind in range(tree_node['start_at'], tree_node['end_at'] + 1):
            for term_ind in idx_term_inds_by_sentence_num[sent_ind]:
                if not term_ind in idx_ATE_term_counts:
                    idx_ATE_term_counts[term_ind] = 0
                idx_ATE_term_counts[term_ind] += 1
        if len(idx_ATE_term_counts) == 0:
            return bool_has_terms


    tree_node['idx_ATE_term_counts'] = idx_ATE_term_counts

    # Add another counter which includes subterms, but with an appropriate weight.
    # Count a little more than half. Want the effect to be felt in rollup.
    # For example, want "tree" in "classification tree" and "regression tree" to count just a little more than the two by themselves combined.
    # FIXME :: Should probably put this weight in in CONFIG.
    subterm_weight = 1 #0.51
    idx_ATE_term_counts_augmented = {term_ind:idx_ATE_term_counts[term_ind] for term_ind in idx_ATE_term_counts}
    for term_ind in idx_ATE_term_counts:
        # Check for subterms
        for subterm_ind in idx_term_ind_to_subterm_inds[term_ind]:
            if not subterm_ind in idx_ATE_term_counts_augmented:
                idx_ATE_term_counts_augmented[subterm_ind] = subterm_weight * idx_ATE_term_counts[term_ind] # Initialize with original term count
            idx_ATE_term_counts_augmented[subterm_ind] += subterm_weight
    tree_node['idx_ATE_term_counts_augmented'] = idx_ATE_term_counts_augmented

    # Recurse and get info from children
    child_nodes_with_no_terms = []
    for child_ind in range(len(tree_node['children'])):
        child_node = tree_node['children'][child_ind]
        bool_child_node_has_terms = recursive_add_term_count_rollups(child_node, idx_term_inds_by_sentence_num, idx_term_ind_to_subterm_inds)

        if not bool_child_node_has_terms:
            child_nodes_with_no_terms.append(child_ind)
            continue

        # idx_ATE_term_counts
        for term_ind in child_node['idx_ATE_term_counts']:
            if not term_ind in idx_ATE_term_counts:
                idx_ATE_term_counts[term_ind] = 0
            idx_ATE_term_counts[term_ind] += child_node['idx_ATE_term_counts'][term_ind]

        # idx_ATE_term_counts_augmented
        for term_ind in child_node['idx_ATE_term_counts_augmented']:
            if not term_ind in idx_ATE_term_counts_augmented:
                idx_ATE_term_counts_augmented[term_ind] = 0
            idx_ATE_term_counts_augmented[term_ind] += child_node['idx_ATE_term_counts_augmented'][term_ind]

    # Remove child nodes with no terms.
    # Have to pop these off in reverse order.
    if len(child_nodes_with_no_terms) > 0:
        child_nodes_with_no_terms.reverse()
        for child_ind in child_nodes_with_no_terms:
            tree_node['children'].pop(child_ind)

    if len(idx_ATE_term_counts) == 0:
        #print(tree_node['node_id'])
        return bool_has_terms
    else:
        bool_has_terms = True

    # Get scaled term counts for use in TF-IDF
    max_term_count = max(tree_node['idx_ATE_term_counts'][term_ind] for term_ind in tree_node['idx_ATE_term_counts'])
    tree_node['idx_term_ind_scaled_freqs'] = {term_ind:tree_node['idx_ATE_term_counts'][term_ind]/max_term_count
                                                for term_ind in tree_node['idx_ATE_term_counts']}

    # Get scaled augmented term counts for use in TF-IDF
    max_term_count = max(tree_node['idx_ATE_term_counts_augmented'][term_ind] for term_ind in tree_node['idx_ATE_term_counts_augmented'])

    # Initialize idx_term_similarities_to_parent_alt_mean, which is populated in another function.
    tree_node['idx_term_similarities_to_parent_alt_mean'] = {term_ind:-1 for term_ind in tree_node['idx_ATE_term_counts']}
    # Initialize idx_tf_idfs, which is populated in another function.
    tree_node['idx_tf_idfs'] = {term_ind:0 for term_ind in tree_node['idx_ATE_term_counts']}

    return bool_has_terms


def recursive_add_terms_and_exemplar_to_tree(tree_node,
                                                embedded_sentences,
                                                term_vector_list,
                                                global_mean_vector=False):

    if global_mean_vector is False:
        global_mean_vector = get_normalized_vector_mean(embedded_sentences)

    # Recurse first.
    for child_node in tree_node['children']:
        recursive_add_terms_and_exemplar_to_tree(child_node,
                                                    embedded_sentences,
                                                    term_vector_list,
                                                    global_mean_vector=global_mean_vector)
    start_at = tree_node['start_at']
    end_at = tree_node['end_at']

    # START CALC NODE "MEAN" -----------------------------------------------------------------------

    node_sentences = embedded_sentences[start_at:end_at + 1]
    node_mean_vector= get_normalized_vector_mean(node_sentences)
    tree_node['mean_vector'] = node_mean_vector

    # Get another version of node_mean_vector which is the average of all child mean vectors (if there are any).
    alt_mean_vector = node_mean_vector
    if len(tree_node['children']) > 0:
        child_mean_vectors = [child_node['alt_mean_vector'] for child_node in tree_node['children']]
        alt_mean_vector = get_normalized_vector_mean(child_mean_vectors)
    tree_node['alt_mean_vector'] = alt_mean_vector

    # END CALC NODE "MEAN" -------------------------------------------------------------------------

    # Sentence exemplar
    inner_product_combinations = get_inner_product_combinations([alt_mean_vector], node_sentences)
    exemplar_ind = start_at + np.flip(np.argsort(inner_product_combinations))[0,0]
    tree_node['exemplar_ind'] = exemplar_ind

    tree_node['sentence_inds_sorted_by_sim_alt_mean_vector'] = start_at + np.flip(np.argsort(inner_product_combinations))[0]

    allowed_term_inds = list(tree_node['idx_ATE_term_counts'].keys())

    # Make sure there's at least one...
    if len(allowed_term_inds) == 0:
        allowed_term_inds = range(len(term_vector_list))

    # Get vectors for allowed terms
    allowed_term_vectors = [term_vector_list[ii] for ii in allowed_term_inds]

    # Find best term for node mean sentence.
    # Also find similarities of all terms to mean and sort.
    inner_product_combinations = get_inner_product_combinations([node_mean_vector], allowed_term_vectors)
    best_match_pos = np.flip(np.argsort(inner_product_combinations))[0,0]
    tree_node['best_term_ind_for_mean'] = allowed_term_inds[best_match_pos]

    idx_term_similarities_to_mean = {}
    for ii in range(len(allowed_term_inds)):
        term_ind = allowed_term_inds[ii]
        sim_to_mean = inner_product_combinations[0, ii]
        idx_term_similarities_to_mean[term_ind] = sim_to_mean
    tree_node['idx_term_similarities_to_mean'] = idx_term_similarities_to_mean

    idx_term_similarities_to_alt_mean = {}
    inner_product_combinations = get_inner_product_combinations([alt_mean_vector], allowed_term_vectors)
    best_match_pos = np.flip(np.argsort(inner_product_combinations))[0,0]
    tree_node['best_term_ind_for_alt_mean'] = allowed_term_inds[best_match_pos]
    for ii in range(len(allowed_term_inds)):
        term_ind = allowed_term_inds[ii]
        sim_to_mean = inner_product_combinations[0, ii]
        idx_term_similarities_to_alt_mean[term_ind] = sim_to_mean
    tree_node['idx_term_similarities_to_alt_mean'] = idx_term_similarities_to_alt_mean


    # Get term similarities to global mean.
    idx_term_similarities_to_global_mean = {}
    inner_product_combinations = get_inner_product_combinations([global_mean_vector], allowed_term_vectors)
    for ii in range(len(allowed_term_inds)):
        term_ind = allowed_term_inds[ii]
        sim_to_mean = inner_product_combinations[0, ii]
        idx_term_similarities_to_global_mean[term_ind] = sim_to_mean
    tree_node['idx_term_similarities_to_global_mean'] = idx_term_similarities_to_global_mean


# Calculate TF-IDF for items in child_node['idx_ATE_term_counts'].
# In this context the "corpus" is the collection of child nodes of tree_node.
# So for tree_node, it calculates TF-IDF for each child relative to their siblings, and sets the values in the child.
def recursive_calc_TF_IDFs_for_child_nodes(tree_node):
    # Do the calculation.
    calc_TF_IDFs_for_child_nodes(tree_node)

    for child_node in tree_node['children']:
        recursive_calc_TF_IDFs_for_child_nodes(child_node)


def recursive_calc_child_term_sims_to_node_mean(tree_node):
    node_alt_mean_vector= tree_node['alt_mean_vector']

    child_term_sims_to_node_mean = []
    tree_node['child_term_sims_to_node_mean'] = child_term_sims_to_node_mean

    for child_node in tree_node['children']:
        child_term_inds = list(child_node['idx_ATE_term_counts'].keys())
        child_term_vectors = [term_vector_list[ii] for ii in child_term_inds]

        inner_product_combinations = get_inner_product_combinations([node_alt_mean_vector], child_term_vectors)
        idx_child_term_similarity_to_mean = {}
        for ii in range(len(child_term_inds)):
            term_ind = child_term_inds[ii]
            sim_to_mean = inner_product_combinations[0, ii]
            idx_child_term_similarity_to_mean[term_ind] = sim_to_mean
        child_term_sims_to_node_mean.append(idx_child_term_similarity_to_mean)
        child_node['idx_term_similarities_to_parent_alt_mean'] = idx_child_term_similarity_to_mean

        # Recurse
        recursive_calc_child_term_sims_to_node_mean(child_node)


def recursive_add_combined_list_of_term_attributes(tree_node):

    # Attributes to use...
    idx_term_similarities_to_mean = tree_node['idx_term_similarities_to_mean']
    idx_term_similarities_to_alt_mean = tree_node['idx_term_similarities_to_alt_mean']
    idx_term_similarities_to_parent_alt_mean = tree_node['idx_term_similarities_to_parent_alt_mean']
    idx_term_similarities_to_global_mean = tree_node['idx_term_similarities_to_global_mean']
    idx_ATE_term_counts = tree_node['idx_ATE_term_counts']
    idx_ATE_term_counts_augmented = tree_node['idx_ATE_term_counts_augmented']
    idx_tf_idfs = tree_node['idx_tf_idfs']

    idx_attributes_by_term = {}
    list_of_attributes_by_term = []

    for term_ind in idx_ATE_term_counts:
        term = terms_list[term_ind]
        sim_to_mean = round(idx_term_similarities_to_mean[term_ind], 4)
        sim_to_alt_mean = round(idx_term_similarities_to_alt_mean[term_ind], 4)
        sim_to_parent_alt_mean = round(idx_term_similarities_to_parent_alt_mean[term_ind], 4)
        sim_to_global_mean = round(idx_term_similarities_to_global_mean[term_ind], 4)
        ATE_count = idx_ATE_term_counts[term_ind]
        ATE_count_augmented = idx_ATE_term_counts_augmented[term_ind]
        tf_idf = round(idx_tf_idfs[term_ind], 4)

        attribute_rec = {
            'term_ind':term_ind,
            'term':term,
            'ATE_count':ATE_count,
            'ATE_count_augmented':ATE_count_augmented,
            'sim_to_mean':sim_to_mean,
            'sim_to_alt_mean':sim_to_alt_mean,
            'sim_to_parent_alt_mean':sim_to_parent_alt_mean,
            'sim_to_global_mean':sim_to_global_mean,
            'tf_idf_rel_siblings':tf_idf
        }

        idx_attributes_by_term[term_ind] = attribute_rec
        list_of_attributes_by_term.append(attribute_rec)

    # Sort list_of_attributes_by_term.
    # DON'T CHANGE THIS UNLESS YOU KNOW WHAT YOU'RE DOING.
    def sort_key_tf_idf(rec):
        sort_by_val_1 = rec['tf_idf_rel_siblings'] * rec['sim_to_alt_mean']
        sort_by_val_2 = rec['sim_to_alt_mean']
        sort_by_val = (sort_by_val_1, sort_by_val_2)
        return sort_by_val

    # Sort list_of_attributes_by_term.
    # DON'T CHANGE THIS UNLESS YOU KNOW WHAT YOU'RE DOING.
    def sort_key_sim_to_alt_mean(rec):
        sort_by_val = rec['sim_to_alt_mean']
        return sort_by_val

    # Sort list_of_attributes_by_term.
    # DON'T CHANGE THIS UNLESS YOU KNOW WHAT YOU'RE DOING.
    max_ATE_count = max(rec['ATE_count'] for rec in list_of_attributes_by_term)
    def sort_key_sim_to_alt_mean_ATE_count(rec):
        sort_by_val = rec['sim_to_alt_mean'] * math.log(rec['ATE_count'])
        return sort_by_val

    # Add sort position to each attribute_rec
    list_of_attributes_by_term = sorted(list_of_attributes_by_term, key=sort_key_tf_idf, reverse=True)
    for ii in range(len(list_of_attributes_by_term)):
        attribute_rec = list_of_attributes_by_term[ii]
        sort_rank_tf_idf = 1 + ii
        attribute_rec['sort_rank_tf_idf'] = sort_rank_tf_idf

    # Add sort position to each attribute_rec
    list_of_attributes_by_term = sorted(list_of_attributes_by_term, key=sort_key_sim_to_alt_mean, reverse=True)
    for ii in range(len(list_of_attributes_by_term)):
        attribute_rec = list_of_attributes_by_term[ii]
        sort_rank_sim_to_alt_mean = 1 + ii
        attribute_rec['sort_rank_sim_to_alt_mean'] = sort_rank_sim_to_alt_mean

    # Add sort position to each attribute_rec
    list_of_attributes_by_term = sorted(list_of_attributes_by_term, key=sort_key_sim_to_alt_mean_ATE_count, reverse=True)
    for ii in range(len(list_of_attributes_by_term)):
        attribute_rec = list_of_attributes_by_term[ii]
        sort_rank_sim_to_alt_mean_ATE_count = 1 + ii
        attribute_rec['sort_rank_sim_to_alt_mean_ATE_count'] = sort_rank_sim_to_alt_mean_ATE_count

    tree_node['idx_attributes_by_term'] = idx_attributes_by_term
    tree_node['list_of_attributes_by_term'] = list_of_attributes_by_term

    # Recurse
    for child_node in tree_node['children']:
        recursive_add_combined_list_of_term_attributes(child_node)


####################################################################################################
####################################################################################################
####################################################################################################


print("Getting text...")
pdf_file_str = 'CraigDuke_Dissertation_Revised.pdf'
# 'McClimon Michael Dissertation.pdf' 'CraigDuke_Dissertation_Revised.pdf' 'Zinser_Dissertation FINAL.pdf'
# '000_Data Jujitsu.pdf' 'TRENCH_REAL_ANALYSIS.PDF' 'Fellowship of the Ring.pdf' 'The Little Schemer.pdf'
# '000_An Introduction to Statistical Learning.pdf' '000_Data Mining.pdf' '000_The Elements of Statistical Learning.pdf'
text = fast_get_text_from_PDF(pdf_file_str)

#--------------------------------------------------------------------------------

print("Spltting text into sentences...")
cache_key = lib_disk_cache.get_hash_key('###SENTENCE_SPLITTINGS### ' + text)
bool_found_in_cache, cached_item = lib_disk_cache.get_it(cache_path, cache_key)
if bool_found_in_cache:
    sentence_list = cached_item
else:
    spacy_client = spacy.load("en_core_web_trf")
    sentence_list = extract_valid_declarative_sentences_from_text(spacy_client, text)
    lib_disk_cache.cache_it(cache_path, cache_key, sentence_list)
    del spacy_client

# Replace text with text from valid sentences.
text = "\n".join(sentence_list)

#--------------------------------------------------------------------------------

# for ii in range(len(sentence_list)):
#     print("\n")
#     print('==================================================================')
#     print(ii)
#     print(sentence_list[ii].replace("\n", " "))
# exit("ABORTING")

#--------------------------------------------------------------------------------

print("Embedding sentences...")
cache_key = lib_disk_cache.get_hash_key('###SENTENCE_EMBEDDINGS### ' + text)
bool_found_in_cache, cached_item = lib_disk_cache.get_it(cache_path, cache_key)
if bool_found_in_cache:
    embedded_sentences = cached_item
else:
    embedded_sentences = l2_normalize(SentenceTransformer('all-mpnet-base-v2').encode(sentence_list))
    lib_disk_cache.cache_it(cache_path, cache_key, embedded_sentences)

#--------------------------------------------------------------------------------

# CONFIG ===========================================================================================

# For term extraction using fast_experimental_modified_CValue() and get_terms_and_counts_from_text().
# BRYAN NOTE :: DON'T CHANGE THIS WITHOUT DOING A LOT OF TESTING!
max_num_words_in_term = 10
min_num_words_in_term = 1
min_freq_count = 2

# For embedding terms using get_embeddings_from_terms_using_context().
# BRYAN NOTE :: DON'T CHANGE THIS WITHOUT DOING A LOT OF TESTING!
term_weight = 10        # 10
sentences_weight = 1     # 1
sentence_window = 0     # 0      # At one point was using max(5, int(len(sentence_list)/100))

# Min # of sentences in sliding window.
# Has to be at least 2 or errors.
min_window_size = 3

# Min # of sentences in tree node.
# min_node_size needs to be at least 1 to avoid errors.
# min_node_size = 1 is TOO SMALL
# BRYAN NOTE :: DON'T CHANGE THIS WITHOUT DOING A LOT OF TESTING!
min_node_size = 2 # 1, 4



#===================================================================================================

print("Getting terms by sentence...")

cache_key = lib_disk_cache.get_hash_key('###sentence_idx_term_counter_list### '
                                            + str(max_num_words_in_term) + '-' + str(min_num_words_in_term) + '-' + str(min_freq_count)
                                            + '-' + text)
bool_found_in_cache, cached_item = lib_disk_cache.get_it(cache_path, cache_key)
if bool_found_in_cache:
    (sentence_idx_term_counter_list, idx_term_counter, idx_terms_to_lemmatized_terms) = cached_item
else:
    sentence_idx_term_counter_list = []
    sentence_idx_terms_to_lemmatized_term_list = []
    spacy_client = spacy.load("en_core_web_sm", disable=["parser", "entity"])
    for ii in range(len(sentence_list)):
        if ii % 100 == 0:
            print("getting terms from sentence #", ii)
        sentence = sentence_list[ii]
        idx_sent_term_counter, \
            _, \
            idx_sent_terms_to_lemmatized_terms = get_terms_and_counts_from_text(sentence, spacy_client=spacy_client,
                                                                                    max_num_words_in_term=max_num_words_in_term,
                                                                                    min_num_words_in_term=min_num_words_in_term,
                                                                                    min_freq_count=1)
        sentence_idx_term_counter_list.append(idx_sent_term_counter)
        sentence_idx_terms_to_lemmatized_term_list.append(idx_sent_terms_to_lemmatized_terms)
    del spacy_client

    # Build idx_term_counter
    idx_term_counter = {}
    for idx_sent_term_counter in sentence_idx_term_counter_list:
        for term in idx_sent_term_counter:
            if not term in idx_term_counter:
                idx_term_counter[term] = 0
            idx_term_counter[term] += idx_sent_term_counter[term]

    # Weed out terms with counts less than min_freq_count
    idx_term_counter = {term:idx_term_counter[term] for term in idx_term_counter if idx_term_counter[term] >= min_freq_count}

    # Build idx_terms_to_lemmatized_terms
    idx_terms_to_lemmatized_terms = {}
    for idx_sent_terms_to_lemmatized_terms in sentence_idx_terms_to_lemmatized_term_list:
        for term in idx_sent_terms_to_lemmatized_terms:
            idx_terms_to_lemmatized_terms[term] = idx_sent_terms_to_lemmatized_terms[term]

    lib_disk_cache.cache_it(cache_path, cache_key, (sentence_idx_term_counter_list, idx_term_counter, idx_terms_to_lemmatized_terms))

# # TESTING
# for term in idx_term_counter:
#     print(term, idx_term_counter[term])
# exit("ABORTING")

print("Getting global term list and counts...")

terms_list = list(idx_term_counter.keys())
term_freq_list = [idx_term_counter[term] for term in terms_list]

#--------------------------------------------------------------------------------


print("# terms before synonyms:", len(terms_list))

print("Finding synonyms stage 1: Using (lower-cased) lemmatized terms.")
idx_lemmatized_term_to_term = {}
for term in idx_terms_to_lemmatized_terms:
    lemmatized_term = idx_terms_to_lemmatized_terms[term]
    if not lemmatized_term in idx_lemmatized_term_to_term:
        idx_lemmatized_term_to_term[lemmatized_term] = []
    if not term in idx_term_counter:
        idx_term_counter[term] = 0
    idx_lemmatized_term_to_term[lemmatized_term].append((term, idx_term_counter[term]))

idx_synonym_to_representative = {}
for lemmatized_term in idx_lemmatized_term_to_term:
    if len(idx_lemmatized_term_to_term[lemmatized_term]) > 1:
        syn_group_rec = idx_lemmatized_term_to_term[lemmatized_term]
        # Sort
        syn_group_rec = sorted(syn_group_rec, key=lambda rec: rec[1], reverse=True)
        repr_term = syn_group_rec[0][0]
        for ii in range(1, len(syn_group_rec)):
            syn_term = syn_group_rec[ii][0]
            idx_synonym_to_representative[syn_term] = repr_term

idx_representative_to_synonyms = {}
for syn_term in idx_synonym_to_representative:
    repr_term = idx_synonym_to_representative[syn_term]
    if not repr_term in idx_representative_to_synonyms:
        idx_representative_to_synonyms[repr_term] = []
    idx_representative_to_synonyms[repr_term].append(syn_term)

# for term in idx_representative_to_synonyms:
#     print(term, ' : ', idx_representative_to_synonyms[term])
# print(len(idx_synonym_to_representative))
# exit("ABORTING")

# Remove non-representative synonyms from terms_list, term_freq_list.
new_terms_list = []
new_term_freq_list = []
for term_ind in range(len(terms_list)):
    term = terms_list[term_ind]
    if term not in idx_synonym_to_representative:
        new_terms_list.append(terms_list[term_ind])
        new_term_freq_list.append(term_freq_list[term_ind])
terms_list = new_terms_list
term_freq_list = new_term_freq_list

print("# terms after stage 1:", len(terms_list))

print("Finding synonyms stage 2: Using sentence context.")
print("Embedding terms stage 2...")
term_vector_list = get_embeddings_from_terms_using_context(terms_list,
                                                            text,
                                                            term_weight,
                                                            sentences_weight,
                                                            sentence_window,
                                                            idx_representative_to_synonyms=idx_representative_to_synonyms)

# Find synonyms
community_list = find_terms_closely_related_by_embedding_sim(term_vector_list)
idx_synonym_to_representative_2 = {}
for community_rec in community_list:
    community_rec = sorted(community_rec, key=lambda term_ind:term_freq_list[term_ind], reverse=True)
    repr_term = terms_list[community_rec[0]]
    for ii in range(1, len(community_rec)):
        syn_term = terms_list[community_rec[ii]]
        idx_synonym_to_representative_2[syn_term] = repr_term

idx_representative_to_synonyms_2 = {}
for syn_term in idx_synonym_to_representative_2:
    repr_term = idx_synonym_to_representative_2[syn_term]
    if not repr_term in idx_representative_to_synonyms_2:
        idx_representative_to_synonyms_2[repr_term] = []
    idx_representative_to_synonyms_2[repr_term].append(syn_term)

# Remove non-representative synonyms from terms_list, term_freq_list.
new_terms_list = []
new_term_freq_list = []
for term_ind in range(len(terms_list)):
    term = terms_list[term_ind]
    if term not in idx_synonym_to_representative_2:
        new_terms_list.append(terms_list[term_ind])
        new_term_freq_list.append(term_freq_list[term_ind])
terms_list = new_terms_list
term_freq_list = new_term_freq_list

print("# terms after stage 2:", len(terms_list))

# Combine idx_representative_to_synonyms and idx_synonym_to_representative_2
idx_representative_to_synonyms_3 = {}
for term in idx_representative_to_synonyms_2:
    syn_terms = set(idx_representative_to_synonyms_2[term])
    for syn_term in idx_representative_to_synonyms_2[term]:
        if syn_term in idx_representative_to_synonyms: # syn_term is a representative term from the first pass...
            syn_terms = syn_terms | set(idx_representative_to_synonyms[syn_term])
    idx_representative_to_synonyms_3[term] = list(syn_terms)
for term in idx_representative_to_synonyms:
    if term in idx_representative_to_synonyms_3:
        idx_representative_to_synonyms_3[term] = list(set(idx_representative_to_synonyms_3[term]) | set(idx_representative_to_synonyms[term]))
    else:
        idx_representative_to_synonyms_3[term] = list(set(idx_representative_to_synonyms[term]))

# Replace the map with the new one.
# At the same time sort everything to make caching work.
idx_representative_to_synonyms = {}
rep_list = sorted(list(idx_representative_to_synonyms_3.keys()))
for rep_term in rep_list:
    idx_representative_to_synonyms[rep_term] = sorted(idx_representative_to_synonyms_3[rep_term])

# Make map of synonyms to representatives
idx_synonym_to_representative = {}
for rep_term in idx_representative_to_synonyms:
    for syn_term in idx_representative_to_synonyms[rep_term]:
        idx_synonym_to_representative[syn_term] = rep_term

#--------------------------------------------------------------------------------

print("# terms final:", len(terms_list))

#--------------------------------------------------------------------------------


print("Embedding terms final...")
term_vector_list = get_embeddings_from_terms_using_context(terms_list,
                                                            text,
                                                            term_weight,
                                                            sentences_weight,
                                                            sentence_window,
                                                            idx_representative_to_synonyms=idx_representative_to_synonyms)

#--------------------------------------------------------------------------------

# Get term ids by term string.
# This does not contain ALL of the original terms, because some of them were removed by the above plural/synonym process.
# The keys are exactly the globally-valid terms.
idx_term_to_term_ind = {}
for term_ind in range(len(terms_list)):
    idx_term_to_term_ind[terms_list[term_ind]] = term_ind

#--------------------------------------------------------------------------------

# Reduce idx_term_counter so it contains only globally-valid terms.
idx_term_counter = {term:idx_term_counter[term] for term in idx_term_counter if term in idx_term_to_term_ind}

idx_term_lower_to_term = {term.lower():term for term in idx_term_to_term_ind}

idx_synonym_lower_to_representative = {syn_term.lower():idx_synonym_to_representative[syn_term] for syn_term in idx_synonym_to_representative}

# Get set of subterms for each term.
# Need to deal with synonyms here.
idx_term_to_subterms = {term:set() for term in idx_term_counter}
for term in idx_term_to_subterms:
    subterms = helper_get_subterms(term, min_num_words_in_term=1)
    for subterm in subterms:
        if subterm.lower() in idx_term_lower_to_term:
            idx_term_to_subterms[term].add(idx_term_lower_to_term[subterm.lower()])
        elif subterm.lower() in idx_synonym_lower_to_representative:
            rep_term = idx_synonym_lower_to_representative[subterm.lower()]
            if rep_term in idx_term_counter:
                idx_term_to_subterms[term].add(rep_term)

# Get list of subterm_inds for each term_ind.
idx_term_ind_to_subterm_inds = {idx_term_to_term_ind[term]:set() for term in idx_term_to_subterms}
for term in idx_term_to_subterms:
    term_ind = idx_term_to_term_ind[term]
    for subterm in idx_term_to_subterms[term]:
        subterm_ind = idx_term_to_term_ind[subterm]
        idx_term_ind_to_subterm_inds[term_ind].add(subterm_ind)

# for term_ind in idx_term_ind_to_subterm_inds:
#     for subterm_ind in idx_term_ind_to_subterm_inds[term_ind]:
#         print(terms_list[term_ind], ':', terms_list[subterm_ind])
# exit("ABORTING")

#--------------------------------------------------------------------------------

# # Examine globally-valid term counts.
# for term in idx_term_to_term_ind:
#     print(term, idx_term_counter[term], idx_term_counts_allowing_subterms[term])
# exit("ABORTING")

#--------------------------------------------------------------------------------

# Use sentence_idx_term_counter_list to build a list of terms by sentence #.
# Deal with synonym info here too.
idx_term_inds_by_sentence_num = {}
for sent_ind in range(len(sentence_idx_term_counter_list)):
    idx_term_inds_by_sentence_num[sent_ind] = set()

    # term might be something that is a synonym and was removed from terms_list...
    for term in sentence_idx_term_counter_list[sent_ind]:
        if term in idx_term_to_term_ind: # Some of them may have been removed by the earlier plural/synonym process.
            term_ind = idx_term_to_term_ind[term]
            idx_term_inds_by_sentence_num[sent_ind].add(term_ind)
        else:
            # Check for synonym
            if term in idx_synonym_to_representative:
                rep_term = idx_synonym_to_representative[term]
                if rep_term in idx_term_to_term_ind:
                    rep_term_ind = idx_term_to_term_ind[rep_term]
                    idx_term_inds_by_sentence_num[sent_ind].add(rep_term_ind)

#--------------------------------------------------------------------------------

num_sentences = len(embedded_sentences)
print("num_sentences:", num_sentences)

#--------------------------------------------------------------------------------

####################################################################################################
# Do the real work here...
# Figure out where the topic is changing...
####################################################################################################

cache_key = lib_disk_cache.get_hash_key('###TOPIC_CHANGE###' + '-' + str(min_window_size) + '-' + str(sentence_list))
bool_found_in_cache, cached_item = lib_disk_cache.get_it(cache_path, cache_key)
if bool_found_in_cache:
    extrema_presence_grid = cached_item
else:
    print("Getting extrema presence grid for topic changes...")
    extrema_presence_grid = get_topic_change_extrema_presence_grid(embedded_sentences, min_window_size)
    # Cache it
    lib_disk_cache.cache_it(cache_path, cache_key, extrema_presence_grid)

# Get counts of extrema by sentence index.
extrema_counts_by_sentence = np.sum(extrema_presence_grid, axis=1)


# Tree ---------------------------------------------------------------------------------------------

tree_root = {
    'level':0,
    'LR':'*',
    'path_to_top':'*',
    'start_at':0,
    'end_at':len(extrema_counts_by_sentence) - 1,
    'children':[]
}
recursive_build_tree_from_extrema_counts(extrema_counts_by_sentence, tree_root, min_node_size)

recursive_remove_useless_children(tree_root)

# FIXME :: DO WE REALLY WANT TO DO THIS?
recursive_expand_children_to_fill_parent(tree_root)

recursive_add_node_ids_and_links_to_parent(tree_root)

# This will remove some nodes which have no valid terms.
recursive_add_term_count_rollups(tree_root, idx_term_inds_by_sentence_num, idx_term_ind_to_subterm_inds)

idx_tree_nodes_by_node_id = get_tree_nodes_by_node_id(tree_root)

# recursive_tree_debug_print(tree_root)
# exit("ABORTING")

recursive_add_terms_and_exemplar_to_tree(tree_root,
                                            embedded_sentences,
                                            term_vector_list)

recursive_calc_TF_IDFs_for_child_nodes(tree_root)

recursive_calc_child_term_sims_to_node_mean(tree_root)

recursive_add_combined_list_of_term_attributes(tree_root)


# # Print leaf nodes
# for node_id in idx_tree_nodes_by_node_id:
#     tree_node = idx_tree_nodes_by_node_id[node_id]
#     if len(tree_node['children']) == 0:
#         print()
#         print_tree_node(tree_node, sentence_list, terms_list, single_node=True)

recursive_tree_print(tree_root, sentence_list, terms_list)

# print_tree_node(idx_tree_nodes_by_node_id[1878], sentence_list, terms_list, single_node=True)
# print_tree_node(idx_tree_nodes_by_node_id[1879], sentence_list, terms_list, single_node=True)
# print_tree_node(idx_tree_nodes_by_node_id[2100], sentence_list, terms_list, single_node=True)

#recursive_tree_print_2(tree_root, sentence_list, terms_list)

exit("ABORTING")




# Graph --------------------------------------------------------------------------------------------

fig, ax= plt.subplots()

y_max = max(window_size_list)
max_extrema_count = max(extrema_counts_by_sentence)

plt.plot([sent_ind for sent_ind in range(len(extrema_counts_by_sentence))],
            extrema_counts_by_sentence*(y_max/max_extrema_count),
            color='#ff7777', zorder=3)

# for window_size in res_by_window_size:
#     y_off = window_size

#     extrema_list = res_by_window_size[window_size]

#     plt.scatter([extrema_ind for extrema_ind in extrema_list],
#                      [y_off for extrema_ind in extrema_list], s=8, c=cm.rainbow([y_off/y_max for extrema_ind in extrema_list]), zorder=4)

plt.show()