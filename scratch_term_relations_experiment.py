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

from lib_corenlp import *
from my_conparse_tree import MyConParseTree
from spacy.language import Language


####################################################################################################
### CONFIG #########################################################################################
####################################################################################################

os.environ["TOKENIZERS_PARALLELISM"] = "false"

cache_path = '/tmp/topic_model_cache'

manual_sentence_boundary_str = '𑗊'

@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == manual_sentence_boundary_str:
            doc[token.i + 1].is_sent_start = True
        else:
            doc[token.i + 1].is_sent_start = False
    return doc

spacy_client = spacy.load("en_core_web_sm")
spacy_client.add_pipe("set_custom_boundaries", before="parser")
spacy_client.add_pipe('benepar', config={'model': 'benepar_en3'})

from allennlp_models.pretrained import load_predictor
coref_predictor = load_predictor("coref-spanbert")

# Params(
#     {

#         'data_loader': {'batch_sampler': {'batch_size': 1, 'sorting_keys': ['text'], 'type': 'bucket'}},
#         'dataset_reader': {'max_sentences': 110, 'max_span_width': 30, 'token_indexers': {'tokens': {'max_length': 512, 'model_name': 'SpanBERT/spanbert-large-cased', 'type': 'pretrained_transformer_mismatched'}}, 'type': 'coref'},
#         'model': {
#             'antecedent_feedforward': {'activations': 'relu', 'dropout': 0.3, 'hidden_dims': 1500, 'input_dim': 9296, 'num_layers': 2}, 'coarse_to_fine': True,
#             'context_layer': {'input_dim': 1024, 'type': 'pass_through'},
#             'feature_size': 20,
#             'inference_order': 2,
#             'initializer': {'regexes': [['.*_span_updating_gated_sum.*weight', {'type': 'xavier_normal'}], ['.*linear_layers.*weight', {'type': 'xavier_normal'}], ['.*scorer.*weight', {'type': 'xavier_normal'}], ['_distance_embedding.weight', {'type': 'xavier_normal'}], ['_span_width_embedding.weight', {'type': 'xavier_normal'}], ['_context_layer._module.weight_ih.*', {'type': 'xavier_normal'}], ['_context_layer._module.weight_hh.*', {'type': 'orthogonal'}]]},
#             'max_antecedents': 50,
#             'max_span_width': 30,
#             'mention_feedforward': {'activations': 'relu', 'dropout': 0.3, 'hidden_dims': 1500, 'input_dim': 3092, 'num_layers': 2},
#             'spans_per_word': 0.4,
#             'text_field_embedder': {'token_embedders': {'tokens': {'max_length': 512, 'model_name': 'SpanBERT/spanbert-large-cased', 'type': 'pretrained_transformer_mismatched'}}},
#             'type': 'coref'
#         },
#         'test_data_path': '/home/dirkg/tank/data/conll12/test.english.v4_gold_conll',
#         'train_data_path': '/home/dirkg/tank/data/conll12/train.english.v4_gold_conll',
#         'trainer': {'learning_rate_scheduler': {'cut_frac': 0.06, 'type': 'slanted_triangular'}, 'num_epochs': 40, 'optimizer': {'lr': 0.0003, 'parameter_groups': [[['.*transformer.*'], {'lr': 1e-05}]], 'type': 'huggingface_adamw'}, 'patience': 10, 'validation_metric': '+coref_f1'},
#         'validation_data_path': '/home/dirkg/tank/data/conll12/dev.english.v4_gold_conll',
#         'validation_dataset_reader': {'max_span_width': 30, 'token_indexers': {'tokens': {'max_length': 512, 'model_name': 'SpanBERT/spanbert-large-cased', 'type': 'pretrained_transformer_mismatched'}}, 'type': 'coref'}
#     }
# )

### END CONFIG #####################################################################################
####################################################################################################



####################################################################################################
####################################################################################################
####################################################################################################


print("Getting text...")
pdf_file_str = '000_An Introduction to Statistical Learning.pdf'
# 'McClimon Michael Dissertation.pdf' 'CraigDuke_Dissertation_Revised.pdf' 'Zinser_Dissertation FINAL.pdf'
# '000_Data Jujitsu.pdf' 'TRENCH_REAL_ANALYSIS.PDF' 'Fellowship of the Ring.pdf' 'The Little Schemer.pdf'
# '000_An Introduction to Statistical Learning.pdf' '000_Data Mining.pdf' '000_The Elements of Statistical Learning.pdf'
text = fast_get_text_from_PDF(pdf_file_str)

#--------------------------------------------------------------------------------

# # TEST FOR WH-word and COREF resolution.
# text = "He threw the book, which is red. These dogs need many cans, which results in us needing lots of food."

sentence_list = fast_split_text_into_sentences(text)
for ii in range(len(sentence_list)):
    sentence_list[ii] = sentence_list[ii].replace("\n", " ")
    sentence_list[ii] = sentence_list[ii].replace("  ", " ")

#--------------------------------------------------------------------------------

# for ii in range(len(sentence_list)):
#     print("\n")
#     print('==================================================================')
#     print(ii)
#     print(sentence_list[ii].replace("\n", " "))
# exit("ABORTING")

#--------------------------------------------------------------------------------

def simple_is_verb_like(pos, deprel):
    return pos in ['VP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] and deprel != 'amod'

def simple_is_wh_word(pos, deprel):
    if pos in ['WDT', 'WP', 'WP$'] and deprel in ['nsubj', 'nmod:poss', 'poss', 'obl', 'obj', 'dobj', 'pobj'] \
            or pos == 'WRB' and deprel == 'advmod':

        return True

    return False

#--------------------------------------------------------------------------------

# You can't use too many sentences, or you can run out of memory.
text = (" " + manual_sentence_boundary_str + " ").join(sentence_list[6000:6100])

doc = spacy_client(text)

predicted_corefs = coref_predictor.predict_tokenized([token.text for token in doc])

#--------------------------------------------------------------------------------

doc_tokens_list = [token for token in doc]

idx_doc_token_info_recs = {}
for token in doc_tokens_list:
    token_i = token.i
    idx_doc_token_info_recs[token_i] = {
        'token_i':token_i,
        'parent_token_i':False,
        'WH_res_token_i':False,
        'coref_res_token_span':False,
        'text':token.text,
        'pos':token.tag_,
        'deprel':token.dep_,
        'lemma':token.lemma_,
        'children':[child_token.i for child_token in token.children]
    }
# Add parents
for token_i in idx_doc_token_info_recs:
    for child_token_i in idx_doc_token_info_recs[token_i]['children']:
        idx_doc_token_info_recs[child_token_i]['parent_token_i'] = token_i

#------------------------------------------------------------------------

# Resolve WH-words
for token_i in idx_doc_token_info_recs:
    token_info = idx_doc_token_info_recs[token_i]
    if simple_is_wh_word(token_info['pos'], token_info['deprel']):
        if token_info['parent_token_i'] is not False:
            parent_token = idx_doc_token_info_recs[token_info['parent_token_i']]
            if parent_token['deprel'] in ['acl:relcl', 'relcl']:
                if parent_token['parent_token_i'] is not False:
                    # Usually the grandparent is the resolution.
                    token_info['WH_res_token_i'] = parent_token['parent_token_i']

#------------------------------------------------------------------------

# Resolve coreferences. Sometimes you have to follow a chain.
idx_coref_res_ref_to_main = {}
coref_clusters = predicted_corefs['clusters']
for cluster in coref_clusters:
    main_item = cluster[0]
    main_item_first_i = main_item[0]
    main_item_last_i = main_item[1]
    referring_items = cluster[1:]
    for referring_item in referring_items:
        referring_item_last_i = referring_item[1]
        idx_coref_res_ref_to_main[referring_item_last_i] = (main_item_first_i, main_item_last_i)

# Follow any resolution chains.
for referring_item_last_i in idx_coref_res_ref_to_main:
    resolution_first_i = idx_coref_res_ref_to_main[referring_item_last_i][0]
    resolution_last_i = idx_coref_res_ref_to_main[referring_item_last_i][1]
    counter = 0 # Want to prevent infinite loops...
    while resolution_last_i in idx_coref_res_ref_to_main:
        counter += 1
        if counter > 1000:
            assert False, "Infinite loop detected in coreferences."
        resolution_first_i = idx_coref_res_ref_to_main[resolution_last_i]
        resolution_last_i = idx_coref_res_ref_to_main[resolution_last_i]

    idx_coref_res_ref_to_main[referring_item_last_i] = (resolution_first_i, resolution_last_i)

for referring_item_last_i in idx_coref_res_ref_to_main:
    idx_doc_token_info_recs[referring_item_last_i]['coref_res_token_span'] = idx_coref_res_ref_to_main[referring_item_last_i]

#------------------------------------------------------------------------

for sent in doc.sents:
    print()
    print(sent)
    for token in sent:
        token_i = token.i
        token_info = idx_doc_token_info_recs[token_i]
        print(token_info)
        if token_info['WH_res_token_i']:
            print("\tWH-RESOLVE TO", token_info['WH_res_token_i'], idx_doc_token_info_recs[token_info['WH_res_token_i']]['text'])
        if token_info['coref_res_token_span']:
            span_start = token_info['coref_res_token_span'][0]
            span_end = token_info['coref_res_token_span'][1]
            coref_res_text = " ".join(idx_doc_token_info_recs[ii]['text'] for ii in range(span_start, span_end + 1))
            print("\tCOREF-RESOLVE TO", token_info['coref_res_token_span'], coref_res_text)

