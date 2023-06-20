from lib_vector_clustering import *
from lib_disk_cache import *
from sentence_transformers import SentenceTransformer


####################################################################################################
### CONFIG #########################################################################################
####################################################################################################


cache_path = '/tmp/topic_model_cache'


### END CONFIG #####################################################################################
####################################################################################################





def get_embeddings_from_terms(terms_list):
    cache_key = lib_disk_cache.get_hash_key('###get_embeddings_from_terms### ' + str(terms_list))
    bool_found_in_cache, cached_item = lib_disk_cache.get_it(cache_path, cache_key)
    if bool_found_in_cache:
        vector_list = cached_item
    else:
        vector_list = l2_normalize(SentenceTransformer('all-mpnet-base-v2').encode(terms_list))
        lib_disk_cache.cache_it(cache_path, cache_key, vector_list)

    return vector_list


def get_embeddings_from_terms_using_context(terms_list, text, term_weight, sentences_weight, sentence_window, idx_representative_to_synonyms=False):
    if idx_representative_to_synonyms is False:
        idx_representative_to_synonyms = {}

    cache_key = lib_disk_cache.get_hash_key('###get_embeddings_from_terms_using_context### '
                                                + str(term_weight) + ' ' + str(sentences_weight)
                                                + str(terms_list) + str(sentence_window) + str(idx_representative_to_synonyms))
    bool_found_in_cache, cached_item = lib_disk_cache.get_it(cache_path, cache_key)
    if bool_found_in_cache:
        vector_list = cached_item
        return vector_list

    sentence_list = fast_split_text_into_sentences(text)
    context_sentences_nums_by_term = {}
    for term in terms_list:
        context_sentences_nums_by_term[term] = []

        # Get sentece #s for this term
        idx_sent_nums = {}
        for sent_num in range(len(sentence_list)):
            sentence = sentence_list[sent_num]
            if sentence.lower().find(term.lower()) > -1:
                idx_sent_nums[sent_num] = True

        # If the term is a synonym, get sentence #s for the representative
        if term in idx_representative_to_synonyms:
            for syn_term in idx_representative_to_synonyms[term]:
                for sent_num in range(len(sentence_list)):
                    sentence = sentence_list[sent_num]
                    if sentence.lower().find(syn_term.lower()) > -1:
                        idx_sent_nums[sent_num] = True

        for sent_num in idx_sent_nums:
            # Look at nearby sentences if supposed to.
            for use_sent_num in range(max(0, sent_num - sentence_window), min(sent_num + 1 + sentence_window, len(sentence_list))):
                context_sentences_nums_by_term[term].append(use_sent_num)


    # Embedding terms
    embedded_terms = l2_normalize(SentenceTransformer('all-mpnet-base-v2').encode(terms_list))

    # Embedding sentences
    embedded_sentences = l2_normalize(SentenceTransformer('all-mpnet-base-v2').encode(sentence_list))

    vector_list = []
    for ii in range(len(embedded_terms)):
        term = terms_list[ii]

        sentence_embeddings_for_term = [embedded_sentences[sent_num] for sent_num in context_sentences_nums_by_term[term]]

        if len(sentence_embeddings_for_term) > 0: # Have some context...

            # Average
            mean_sentence_embedding = get_normalized_vector_mean(sentence_embeddings_for_term)

            # Make new vector for term, normalize!
            vector = l2_normalize(term_weight * embedded_terms[ii] + sentences_weight * mean_sentence_embedding)

            # Add to new vector list, and add term to filtered terms list.
            vector_list.append(vector)

        else: # Just use the term vector itself.
            vector_list.append(embedded_terms[ii])

    vector_list = np.array(vector_list)

    lib_disk_cache.cache_it(cache_path, cache_key, vector_list)

    return vector_list


def recursive_PCA_plus_minus_print(PCA_space_vect_list, terms_list, plus_minus_list_list, coord_plus_minus_path, max_depth=2):

    if len(coord_plus_minus_path) >= max_depth:
        return

    mean = np.mean(plus_minus_list_list[:, len(coord_plus_minus_path)])
    std = np.std(plus_minus_list_list[:, len(coord_plus_minus_path)])

    u_cutoff = mean + 1*std
    l_cutoff = mean - 1*std

    plus_list = plus_minus_list_list[plus_minus_list_list[:, len(coord_plus_minus_path)] > u_cutoff]
    if len(plus_list) > 0:
        plus_mean = get_normalized_vector_mean(plus_list)
        inner_product_combinations = get_inner_product_combinations([plus_mean], PCA_space_vect_list)
        plus_exemplar_ind = np.flip(np.argsort(inner_product_combinations))[0,0]

        new_coord_plus_minus_path = coord_plus_minus_path[:]
        new_coord_plus_minus_path.append(1)
        print("\t"*len(coord_plus_minus_path), terms_list[plus_exemplar_ind], new_coord_plus_minus_path)
        recursive_PCA_plus_minus_print(PCA_space_vect_list, terms_list, plus_list, new_coord_plus_minus_path, max_depth=max_depth)

    zero_list = plus_minus_list_list[plus_minus_list_list[:, len(coord_plus_minus_path)] >= l_cutoff]
    zero_list = zero_list[zero_list[:, len(coord_plus_minus_path)] <= u_cutoff]
    if len(zero_list) > 0:
        zero_mean = get_normalized_vector_mean(zero_list)
        inner_product_combinations = get_inner_product_combinations([zero_mean], PCA_space_vect_list)
        zero_exemplar_ind = np.flip(np.argsort(inner_product_combinations))[0,0]

        new_coord_plus_minus_path = coord_plus_minus_path[:]
        new_coord_plus_minus_path.append(0)
        print("\t"*len(coord_plus_minus_path), terms_list[zero_exemplar_ind], new_coord_plus_minus_path)
        recursive_PCA_plus_minus_print(PCA_space_vect_list, terms_list, zero_list, new_coord_plus_minus_path, max_depth=max_depth)

    minus_list = plus_minus_list_list[plus_minus_list_list[:, len(coord_plus_minus_path)] < l_cutoff]
    if len(minus_list) > 0:
        minus_mean = get_normalized_vector_mean(minus_list)
        inner_product_combinations = get_inner_product_combinations([minus_mean], PCA_space_vect_list)
        minus_exemplar_ind = np.flip(np.argsort(inner_product_combinations))[0,0]

        new_coord_plus_minus_path = coord_plus_minus_path[:]
        new_coord_plus_minus_path.append(-1)
        print("\t"*len(coord_plus_minus_path), terms_list[minus_exemplar_ind], new_coord_plus_minus_path)
        recursive_PCA_plus_minus_print(PCA_space_vect_list, terms_list, minus_list, new_coord_plus_minus_path, max_depth=max_depth)




def examine_PCA_stuff(terms_list, text, max_depth=5):
    term_weight = 2
    sentences_weight = 1
    sentence_window = 0
    vector_list = get_embeddings_from_terms_using_context(terms_list, text, term_weight, sentences_weight, sentence_window)


    # TESTING: See what happens if take window averages before PCA...
    window_size = 101
    vector_array = np.array(vector_list)
    mean_vectors_array = np.zeros((vector_array.shape[0] + 1 - window_size, vector_array.shape[1]))
    for ii in range(vector_array.shape[0] + 1 - window_size):
        mean_vectors_array[ii] = np.mean(vector_array[ii:ii+window_size, :], axis=0)
    vector_list = mean_vectors_array

    pca = sklearn_PCA(n_components = 0.9999, whiten=True).fit(vector_list)

    # Transform stuff to PCA component space
    PCA_space_vect_list = l2_normalize(pca.transform(vector_list))

    recursive_PCA_plus_minus_print(PCA_space_vect_list, terms_list, PCA_space_vect_list, [], max_depth=max_depth)



