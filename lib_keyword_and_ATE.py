import spacy
from spacy.matcher import Matcher
import pytextrank
from collections import defaultdict
import lib_disk_cache
from lib_book_corpus import *
import math

####################################################################################################
####################################################################################################
####################################################################################################

default_spacy_model_str = 'en_core_web_trf'   # 'en_core_web_trf' 'en_core_web_sm'



# UPDATE 2022-04-11 BRYAN REVISED PATTERNS
class MatcherPatterns:

    # Parts of patterns --------------------------------------------------------

    hyphen_zero_or_more = {
                "TAG": {
                    "IN": ["HYPH"]
                },
                "OP": "*"
            }

    number_zero_or_more = {
                "POS": {
                    "IN": ["NUM"]
                },
                "OP": "*"
            }


    non_punct_noun_one_or_more = {
                "POS": {
                    "IN": ["NOUN", "PROPN"]
                },
                "OP": "+",
                "IS_PUNCT": False
            }

    noun_one_or_more = {
                "POS": {
                    "IN": ["NOUN", "PROPN"]
                },
                "OP": "+"
            }

    adj_or_noun_zero_or_more = {
                "POS": {
                    "IN": ["ADJ", "NOUN", "PROPN"]
                },
                "OP": "*"
            }

    verb_gerund_zero_or_more = {
                "POS": {
                    "IN": ["VERB"]
                },
                "TAG": {
                    "IN": ["VBG"]
                },
                "OP": "*"
            }

    verb_gerund_one_or_more = {
                "POS": {
                    "IN": ["VERB"]
                },
                "TAG": {
                    "IN": ["VBG"]
                },
                "OP": "+"
            }

    verb_zero_or_more = {
                "POS": {
                    "IN": ["VERB"]
                },
                "OP": "*"
            }


    not_adj_or_noun = {
                "POS": {
                    "NOT_IN": ["ADJ", "NOUN", "PROPN"]
                }
            }

    # "'s, not"
    particle = {
                "POS": "PART",
                "OP": "+"
            }

    adposition_det_conj = {
                "POS": {
                    "IN": ["ADP", "DET", "CCONJ"]
                },
                "OP": "+"
            }

    not_adj_noun_or_adposition = {
                "POS": {
                    "NOT_IN": ["ADJ", "NOUN", "PROPN", "ADP"]
                }
            }


    # Patterns -----------------------------------------------------------------

    # pattern_0 = [
    #         adj_or_noun_zero_or_more,
    #         non_punct_noun_one_or_more,
    #         not_adj_or_noun
    #     ]

    pattern_1 = [
            adj_or_noun_zero_or_more,
            verb_gerund_zero_or_more,
            number_zero_or_more,
            hyphen_zero_or_more,
            adj_or_noun_zero_or_more,
            verb_gerund_zero_or_more,
            verb_zero_or_more,
            number_zero_or_more,
            hyphen_zero_or_more,
            adj_or_noun_zero_or_more,
            verb_gerund_zero_or_more,
            verb_zero_or_more,
            non_punct_noun_one_or_more,
            not_adj_or_noun
        ]

    pattern_1A = [
            adj_or_noun_zero_or_more,
            verb_gerund_zero_or_more,
            verb_zero_or_more,
            number_zero_or_more,
            hyphen_zero_or_more,
            adj_or_noun_zero_or_more,
            verb_gerund_zero_or_more,
            verb_zero_or_more,
            number_zero_or_more,
            hyphen_zero_or_more,
            adj_or_noun_zero_or_more,
            verb_gerund_one_or_more,
            verb_zero_or_more,
            not_adj_or_noun
        ]

    pattern_2 = [
            non_punct_noun_one_or_more,

            particle,

            adj_or_noun_zero_or_more,
            non_punct_noun_one_or_more,

            not_adj_or_noun
        ]

    pattern_3 = [
            adj_or_noun_zero_or_more,
            non_punct_noun_one_or_more,

            adposition_det_conj,

            adj_or_noun_zero_or_more,
            non_punct_noun_one_or_more,

            not_adj_noun_or_adposition
        ]

    # Thing to give to SpaCy ---------------------------------------------------

    # The pattern to check against for filtering term candidates.
    #   See Ahrenberg, L. (2009). Term extraction : A Review Draft Version 091221.
    patterns = [
        #pattern_0,

        pattern_1,

        pattern_1A,

        pattern_2,

        pattern_3

    ]

# # THE ORIGINAL PATTERNS FROM PyATE.
# class MatcherPatterns:
#     noun, adj, prep = (
#         {
#             "POS": "NOUN",
#             "IS_PUNCT": False
#         },
#         {
#             "POS": "ADJ",
#             "IS_PUNCT": False
#         },
#         {
#             "POS": "ADP",
#             "IS_PUNCT": False
#         },
#     )

#     # The pattern to check against for filtering term candidates.
#     #   See Ahrenberg, L. (2009). Term extraction : A Review Draft Version 091221.
#     patterns = [
#         [adj],
#         [{
#             "POS": {
#                 "IN": ["ADJ", "NOUN"]
#             },
#             "OP": "*",
#             "IS_PUNCT": False
#         }, noun],
#         [
#             {
#                 "POS": {
#                     "IN": ["ADJ", "NOUN"]
#                 },
#                 "OP": "*",
#                 "IS_PUNCT": False
#             },
#             noun,
#             prep,
#             {
#                 "POS": "DET",
#                 "OP": "?",
#                 "IS_PUNCT": False
#             },
#             {
#                 "POS": {
#                     "IN": ["ADJ", "NOUN"]
#                 },
#                 "OP": "*",
#                 "IS_PUNCT": False
#             },
#             noun,
#         ],
#     ]


def count_words_in_string(s):
    return len(s.split())


def get_lemmas_from_text(text, spacy_client=False):
    if spacy_client is False:
        spacy_client = spacy.load(default_spacy_model_str, disable=["parser", "entity"])

    idx_lemmas = {}

    doc = spacy_client(text)
    for token in doc:
        word = token.text.strip().replace('\n', ' ')
        pos = token.pos_
        word_with_POS = (word, pos)
        lemma = token.lemma_
        idx_lemmas[word_with_POS] = lemma

    return idx_lemmas


def get_terms_and_counts_from_text(text,
                                    spacy_client=False,
                                    max_num_words_in_term=4,
                                    min_num_words_in_term=1,
                                    min_freq_count=1,
                                    idx_term_filter=False,
                                    force_lowercase_lemmas=True):
    if spacy_client is False:
        spacy_client = spacy.load(default_spacy_model_str, disable=["parser", "entity"])
        spacy_client.max_length = 2 * len(text)
    doc = spacy_client(text)

    idx_term_counter = defaultdict(int)
    idx_lemmatized_term_counter = defaultdict(int)
    idx_terms_to_lemmatized_terms = {}

    def add_to_counter(matcher, doc, i, matches):
        match_id, start, end = matches[i]
        #term = str(doc[start:end])
        term = str(doc[start:end - 1]) # UPDATE BRYAN :: I changed the patterns so they always include non-noun at the end. Strip it off!

        # Bail if the term ends with something other than a letter
        term = term.strip()
        #good_ending = re.compile(r"[a-zA-Z]$")
        good_ending = re.compile(r"[^\W\d_]$")
        if not good_ending.search(term):
            return

        # Figure out if has a proper noun
        idx_POS = {token.pos_:True for token in doc[start:end - 1]}
        is_proper_noun = 'PROPN' in idx_POS

        if not is_proper_noun:
            # Lower-case it in case it's not already...
            term = term.lower()

        # Get lemmatized version of term.
        lemmatized_term = ' '.join(token.lemma_ for token in doc[start:end - 1])
        if not is_proper_noun or force_lowercase_lemmas:
            # Lower-case it in case it's not already...
            lemmatized_term = lemmatized_term.lower()
        # Remove non-alpha characters.
        lemmatized_term = re.sub('[^a-zA-z]', ' ', lemmatized_term)
        lemmatized_term = re.sub(' +', ' ', lemmatized_term)
        # Strip whitespace.
        lemmatized_term = lemmatized_term.replace("\n", " ").strip()

        if not idx_term_filter or term in idx_term_filter:
            num_words = count_words_in_string(term)
            if min_num_words_in_term <= num_words and num_words <= max_num_words_in_term:
                idx_term_counter[term] += 1
                idx_lemmatized_term_counter[lemmatized_term] += 1
                 # Add lemma to XREF
                idx_terms_to_lemmatized_terms[term] = lemmatized_term

    # Perform pattern matching...
    spacy_matcher = Matcher(spacy_client.vocab)
    for ii, pattern in enumerate(MatcherPatterns.patterns):
        spacy_matcher.add("term{}".format(ii), [pattern],
                        on_match=add_to_counter,
                        greedy='LONGEST')
    spacy_matcher(doc)

    idx_term_counter = {term:idx_term_counter[term] for term in idx_term_counter \
                            if idx_lemmatized_term_counter[idx_terms_to_lemmatized_terms[term]] >= min_freq_count}
    idx_lemmatized_term_counter = {term:idx_lemmatized_term_counter[term] for term in idx_lemmatized_term_counter \
                            if idx_lemmatized_term_counter[term] >= min_freq_count}


    return idx_term_counter, idx_lemmatized_term_counter, idx_terms_to_lemmatized_terms

# This figures out how "spread out" a term is when it appears as a suffix of other terms.
# Want to capture the ideas that:
#   1) If a term appears only as a suffix of one other distinct term,
#           then the term is likely not a good one.
#       Example: "vector machine" in "support vector machine"
#   2) On the other hand, if a term appears as a suffix of many other distinct terms,
#           then maybe the term can stand on its own since may capture a commonality somehow.
#       Example: "swim meet", "track meet", "gymnastics meet", etc...
#
# Returns a dictionary of tuples (suffix_count, suffix_factor), indexed by term where:
#   1) suffix_count is the frequency (# of times in the text) the term appears as a suffix of another term with one additional word.
#           = SUM(freq term_2 where term_2 has one more word than term and term is a suffix of term_2) # "roses" in "red roses"
#   2) suffix_factor is the maximum ratio of (freq term_2)/suffix_count, where term_2 is as above.
#           If the term never appears as a suffix (so suffix_count is 0),
#               then suffix_factor is set to 0 (kind of arbitrary).
#           The idea is that suffix_factor close to 0 might indicate the term could stand on its own very well,
#               but suffix_factor close to 1, on the other hand, might indicate that it does not.
# This calculation doesn't take into account the number of times a term occurs in text as a standalone term, i.e. NOT as a suffix of another term.
#
# EXAMPLE:
#       "The red blue dog jumps over the yellow blue dog, but his dog likes another dog, and the red blue dog."
#   gives
#       red blue dog (0, False)
#       blue dog (3, 0.6666666666666666)
#       dog (4, 0.75)
#       yellow blue dog (0, False)
#       his dog (0, False)
#
# idx_term_counter should be as returned by idx_term_counter.
#
# UPDATE:
#       There is a "double-dipping" bug resulting from the fact that two different terms returned by
#       the matcher in get_terms_and_counts_from_text() in a previous step
#       can both yield here the same suffix, even when they occur in the same text position.
#       Example:
#               "It is Mike’s fault"
#           The matcher gives these terms:
#               "fault"
#               "'s fault"
#               "Mike's fault"
#           This means that the term "fault" will have suffix_count = 2, but a term count of only 1.
#       FIX:
#           I am not going to try to fix the issue at the root-cause level.
#           Instead, I'm going to put two Band-Aids on the situation:
#               1) Don't allow suffix_count to be greater than the term frequency.
#               2) Try to patch up the matching patterns so this doesn't happen.
def get_suffix_factors(idx_term_counter):
    idx_occurs_as_substring_details = {}
    for superterm in idx_term_counter:
        term = ' '.join(superterm.split(' ')[1:])
        if term in idx_term_counter:
            if not term in idx_occurs_as_substring_details:
                idx_occurs_as_substring_details[term] = {}
            idx_occurs_as_substring_details[term][superterm] = idx_term_counter[superterm]

    # print(idx_occurs_as_substring_details)
    #   Example:
    #       "The red blue dog jumps over the yellow blue dog, but his dog likes another dog, and the red blue dog."
    #   gives
    #       idx_occurs_as_substring_details = {'blue dog': {'red blue dog': 2, 'yellow blue dog': 1}, 'dog': {'blue dog': 3, 'his dog': 1}}

    idx_suffix_factors = {}

    for term in idx_term_counter:
        suffix_count = 0
        suffix_factor = 1
        if term in idx_occurs_as_substring_details:
            suffix_count = sum(idx_occurs_as_substring_details[term][superterm] for superterm in idx_occurs_as_substring_details[term])
            if suffix_count > idx_term_counter[term]:
                suffix_count = idx_term_counter[term]
            suffix_factor = max(min(suffix_count,idx_occurs_as_substring_details[term][superterm])/suffix_count for superterm in idx_occurs_as_substring_details[term])
        idx_suffix_factors[term] = (suffix_count, suffix_factor)

    return idx_suffix_factors


def helper_get_subterms(s, min_num_words_in_term=1):
    words = s.split(' ')
    if len(words) <= min_num_words_in_term:
        return []
    subsequences = []
    for left in range(len(words) + 1):
        for right in range(left + 1, len(words) + 1):
            if left == 0 and right == len(words):
                continue
            subsequences.append(' '.join(words[left:right]))
    return subsequences


# Returns list of [term, ranking, ComboBasic score, term count in text].
# Results sorted by score descending.
# To match a (bug-corrected) SpaCy PyATE version, convert the text to lower case before running.
def fast_ComboBasic(text, max_num_words_in_term=4, min_num_words_in_term=1, spacy_client=False, min_freq_count=1, idx_term_filter=False, lemmatized=False):
    if spacy_client is False:
        spacy_client = spacy.load(default_spacy_model_str, disable=["parser", "entity"])
        spacy_client.max_length = 2 * len(text)

    idx_term_counter, \
    idx_lemmatized_term_counter, \
    idx_terms_to_lemmatized_terms = get_terms_and_counts_from_text(text,
                                                        spacy_client=spacy_client,
                                                        max_num_words_in_term=max_num_words_in_term,
                                                        min_num_words_in_term=min_num_words_in_term,
                                                        min_freq_count=min_freq_count,
                                                        idx_term_filter=idx_term_filter)

    if lemmatized:
        idx_term_counter = idx_lemmatized_term_counter

    idx_occurs_as_substring_counter = {term:0 for term in idx_term_counter}
    idx_occurs_as_superstring_counter = {term:0 for term in idx_term_counter}

    for term in idx_term_counter:
        sub_terms = helper_get_subterms(term, min_num_words_in_term=min_num_words_in_term)
        for sub_term in sub_terms:
            if sub_term in idx_term_counter:
                idx_occurs_as_superstring_counter[term] += 1
                idx_occurs_as_substring_counter[sub_term] += 1

    combo_basics_algorithm_weights = [1, 0.75, 0.1]
    c = combo_basics_algorithm_weights

    candidate_terms_list = []
    for term in idx_term_counter:
        score = c[0] * count_words_in_string(term) * math.log(idx_term_counter[term]) \
                + c[1] * idx_occurs_as_substring_counter[term] \
                + c[2] * idx_occurs_as_superstring_counter[term]
        candidate_terms_list.append([term, 0, score, idx_term_counter[term]])

    candidate_terms_list = sorted(candidate_terms_list, key=lambda rec: (rec[2], rec[0]), reverse=True)

    ranking = 0
    for rec in candidate_terms_list:
        ranking += 1
        rec[1] = ranking

    return candidate_terms_list


def fast_ComboBasic_corpus(doc_list, max_num_words_in_term=4, min_num_words_in_term=1, spacy_client=False, min_freq_count=1, idx_term_filter=False, lemmatized=False):
    if spacy_client is False:
        spacy_client = spacy.load(default_spacy_model_str, disable=["parser", "entity"])
        max_text_len = max(len(doc_text) for doc_text in doc_list)
        spacy_client.max_length = 2 * max_text_len

    candidate_terms_list_by_doc = []
    for doc_text in doc_list:
        candidate_terms_list_by_doc.append(fast_ComboBasic(doc_text,
                                                            max_num_words_in_term=max_num_words_in_term,
                                                            min_num_words_in_term=min_num_words_in_term,
                                                            spacy_client=spacy_client,
                                                            min_freq_count=min_freq_count,
                                                            idx_term_filter=idx_term_filter,
                                                            lemmatized=lemmatized))

    return candidate_terms_list_by_doc


# Returns list of [term, ranking, tf_idf, tf, # of docs present].
# Results sorted by tf_idf descending.
def fast_TF_IDF_data(doc_list, max_num_words_in_term=4, min_num_words_in_term=1, spacy_client=False, min_freq_count=1, idx_term_filter=False, lemmatized=False):
    if spacy_client is False:
        spacy_client = spacy.load(default_spacy_model_str, disable=["parser", "entity"])
        max_text_len = max(len(doc_text) for doc_text in doc_list)
        spacy_client.max_length = 2 * max_text_len

    tfs_by_doc = []
    idx_corpus_doc_presence_counts = defaultdict(int)
    for doc_text in doc_list:
        idx_term_counter, \
        idx_lemmatized_term_counter, \
        idx_terms_to_lemmatized_terms = get_terms_and_counts_from_text(doc_text,
                                                            spacy_client=spacy_client,
                                                            max_num_words_in_term=max_num_words_in_term,
                                                            min_num_words_in_term=min_num_words_in_term,
                                                            min_freq_count=min_freq_count,
                                                            idx_term_filter=idx_term_filter)

        if lemmatized:
            idx_term_counter = idx_lemmatized_term_counter

        # Compute term frequencies for this document.
        total_term_count = sum(idx_term_counter[term] for term in idx_term_counter )
        tfs_by_doc.append({term: idx_term_counter[term]/ total_term_count for term in idx_term_counter})

        # Add contribution to document presence counts.
        for term in idx_term_counter:
            idx_corpus_doc_presence_counts[term] += 1

    idx_corpus_doc_presence_counts = dict(idx_corpus_doc_presence_counts)

    # Compute TF-IDF by document
    num_docs = len(doc_list)
    tf_idfs_by_doc = []
    for doc_tfs in tfs_by_doc:
        tf_idfs_by_doc.append({term: doc_tfs[term] * math.log(num_docs/idx_corpus_doc_presence_counts[term]) for term in doc_tfs})

    # Compute return value which has everything in it...
    tf_idf_recs_by_doc = []
    for ii in range(len(tf_idfs_by_doc)):
        doc_tf_idfs = tf_idfs_by_doc[ii]
        doc_tf_idf_recs = list([term, 0, doc_tf_idfs[term], tfs_by_doc[ii][term], idx_corpus_doc_presence_counts[term]] for term in doc_tf_idfs)
        doc_tf_idf_recs = sorted(doc_tf_idf_recs, key = lambda rec: (rec[2], rec[0]), reverse=True)
        ranking = 0
        for rec in doc_tf_idf_recs:
            ranking += 1
            rec[1] = ranking
        tf_idf_recs_by_doc.append(doc_tf_idf_recs)


    return tf_idf_recs_by_doc, idx_corpus_doc_presence_counts


# Returns list of [term, ranking, score, term count in text, standalone count].
# Results sorted by score descending.
# Attempts to severely punish terms which appear only as a suffix of one distinct term,
#   but less severely punish terms which appear as suffixes of many different terms.
#   The idea is that if a term appears as a suffix of many different terms, then it may capture some commonality.
def fast_experimental_modified_CValue(text,
                                        max_num_words_in_term=4,
                                        min_num_words_in_term=1,
                                        spacy_client=False,
                                        min_freq_count=1,
                                        idx_term_filter=False,
                                        lemmatized=False,
                                        weights=[0.1, 1, .5]):
    if spacy_client is False:
        spacy_client = spacy.load(default_spacy_model_str, disable=["parser", "entity"])
        spacy_client.max_length = 2 * len(text)

    idx_term_counter, \
    idx_lemmatized_term_counter, \
    idx_terms_to_lemmatized_terms = get_terms_and_counts_from_text(text,
                                                        spacy_client=spacy_client,
                                                        max_num_words_in_term=max_num_words_in_term,
                                                        min_num_words_in_term=min_num_words_in_term,
                                                        min_freq_count=min_freq_count,
                                                        idx_term_filter=idx_term_filter)

    if lemmatized:
        idx_term_counter = idx_lemmatized_term_counter

    idx_suffix_factors = get_suffix_factors(idx_term_counter)

    candidate_terms_list = []
    for term in idx_term_counter:
        suffix_count = idx_suffix_factors[term][0]
        suffix_factor = idx_suffix_factors[term][1]

        # suffix_factor close to 0 means term might stand alone well,
        #   but close to 1 means might not stand alone well.

        term_num_words = count_words_in_string(term)
        term_freq = idx_term_counter[term]

        standalone_freq = term_freq - suffix_count

        alpha = weights[0]
        beta = weights[1]
        gamma = weights[2]

        # alpha should probably be in the range (0.1, infty), although two alpha values above 2 or so probably return similar results.
        assert alpha > 0
        weighted_freq = beta * standalone_freq + gamma * suffix_count * (1 - suffix_factor)
        score = math.log(alpha + term_num_words) * (weighted_freq)

        candidate_terms_list.append([term, 0, score, idx_term_counter[term], standalone_freq])

    candidate_terms_list = sorted(candidate_terms_list, key=lambda rec: (rec[2], rec[0]), reverse=True)

    ranking = 0
    for rec in candidate_terms_list:
        ranking += 1
        rec[1] = ranking

    return candidate_terms_list


def fast_experimental_modified_CValue_corpus(doc_list,
                                                max_num_words_in_term=4,
                                                min_num_words_in_term=1,
                                                spacy_client=False,
                                                min_freq_count=1,
                                                idx_term_filter=False,
                                                lemmatized=False,
                                                weights=[0.1, 1, .5]):
    if spacy_client is False:
        spacy_client = spacy.load(default_spacy_model_str, disable=["parser", "entity"])
        max_text_len = max(len(doc_text) for doc_text in doc_list)
        spacy_client.max_length = 2 * max_text_len

    candidate_terms_list_by_doc = []
    for doc_text in doc_list:
        candidate_terms_list_by_doc.append(fast_experimental_modified_CValue(doc_text,
                                                            max_num_words_in_term=max_num_words_in_term,
                                                            min_num_words_in_term=min_num_words_in_term,
                                                            spacy_client=spacy_client,
                                                            min_freq_count=min_freq_count,
                                                            idx_term_filter=idx_term_filter,
                                                            lemmatized=lemmatized,
                                                            weights=weights))

    return candidate_terms_list_by_doc

####################################################################################################


# # REFERENCE IMPLEMENTATION
# def spacy_reference_ComboBasic(text, max_num_words_in_term=4, allow_single_word_terms=True, spacy_client=False):
#     if spacy_client is False:
#         spacy_client = spacy.load(default_spacy_model_str, disable=["parser", "entity"])
#         spacy_client.max_length = 2 * len(text)
#         pyate.term_extraction.TermExtraction.config["MAX_WORD_LENGTH"] = max_num_words_in_term

#     combo_basics_algorithm_weights = [1, 0.75, 0.1]

#     res_combo_basic = dict(pyate.combo_basic(text,
#                                                 nlp=spacy_client,
#                                                 weights=combo_basics_algorithm_weights,
#                                                 have_single_word=allow_single_word_terms).sort_values(ascending=False))

#     candidate_terms_list = list([term, 0, res_combo_basic[term]] for term in res_combo_basic)
#     # Sort
#     candidate_terms_list = sorted(candidate_terms_list, key=lambda rec: (rec[2], rec[0]), reverse=True)

#     ranking = 0
#     for rec in candidate_terms_list:
#         ranking += 1
#         rec[1] = ranking

#     return candidate_terms_list


# # REFERENCE IMPLEMENTATION
# def spacy_reference_TextRank(text):
#     spacy_client = spacy.load(default_spacy_model_str)
#     spacy_client.add_pipe("textrank")

#     chunk_length = 500000 # SpaCy will complain if longer than 1 million characters.

#     text_chunks = []
#     while(len(text) > chunk_length):
#         text_chunk = text[:chunk_length]
#         text_chunks.append(text_chunk)
#         text = text[chunk_length:]
#     if len(text) > 0:
#         text_chunks.append(text)

#     # Choose the maximum score of any term that appears in any chunk.
#     idx_best_scores_by_term = {}
#     for text_chunk in text_chunks:
#         chunk_results = []
#         doc = spacy_client(text_chunk)
#         for phrase in doc._.phrases:
#             term = phrase.text.lower().strip()
#             score = phrase.rank
#             if not term in idx_best_scores_by_term:
#                 idx_best_scores_by_term[term] = score
#             elif score > idx_best_scores_by_term[term]:
#                 idx_best_scores_by_term[term] = score

#     candidate_terms_list = list([term, 0, idx_best_scores_by_term[term]] for term in idx_best_scores_by_term)
#     # Sort
#     candidate_terms_list = sorted(candidate_terms_list, key=lambda rec: rec[2], reverse=True)

#     ranking = 0
#     for rec in candidate_terms_list:
#         ranking += 1
#         rec[1] = ranking

#     return candidate_terms_list


####################################################################################################
####################################################################################################
####################################################################################################

# # TESTING

# text = "It turns out that it is possible to extend the two-class logistic regression approach to the setting of K > 2 classes."

# spacy_client = spacy.load(default_spacy_model_str, disable=["parser", "entity"])
# doc = spacy_client(text)

# for token in doc:
#     print(token, token.pos_, token.tag_, token.is_punct)

# idx_term_counter, idx_lemmatized_term_counter, idx_terms_to_lemmatized_terms = get_terms_and_counts_from_text(text, max_num_words_in_term = 10)

# print()

# for term in idx_term_counter:
#     term_count = idx_term_counter[term]
#     lemmatized_term = idx_terms_to_lemmatized_terms[term]
#     lemmatized_count = idx_lemmatized_term_counter[lemmatized_term]
#     print(term)
