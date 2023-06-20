# https://spacy.io/usage/linguistic-features
# https://spacy.io/usage/processing-pipelines
# https://spacy.io/universe/project/self-attentive-parser

import coreferee
import benepar, spacy, os
import re

print("STARTING...")

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
nlp.add_pipe('coreferee')

text = 'My sister has a horse and a red dog. She loves him, and he loves her.'
doc = nlp(text)

# doc[0]._.coref_chains.print()

for sent in doc.sents:
    for thing in sent:
        print(thing.i)
        print(thing.tag_, thing.text)
        print(thing._.coref_chains)
        print(doc._.coref_chains.resolve(doc[thing.i]))
        print()

#sent_tokens = list(token for token in sent)

# print(dir(sent))
# print(dir(sent._))

# for noun_chunk in sent.noun_chunks:
#     print(noun_chunk)

# print(sent.text)
# print(sent._.parse_string)
#print(sent._.labels)
# print("SENTENCE TOKENS")
# sent_tokens = list(token for token in sent)
# for token in sent_tokens:
#     # print(dir(token))
#     print("\t", token.text, token.lemma_, token.tag_, token.pos_, token.dep_, token.i, '#' + token.whitespace_ + '#')
#     for item in token.children:
#         print("\t\t", item)

# print(sent.text)
# print(sent._.parse_string)
# print(sent._.labels)
# print("CHILDREN:")
# for child in sent._.children:
#     print("\t", child.text, child._.labels, child._.parse_string)


# def show_spacy_const_tree(spacy_tree, level=0):
#     print(type(spacy_tree))
#     print("\t"*level, spacy_tree._.labels, spacy_tree.text, spacy_tree._.parse_string)
#     print("\t"*level, spacy_tree.start, spacy_tree.end)
#     if len(list(spacy_tree._.children)) == 0:
#         parse_str = spacy_tree._.parse_string
#         space_pos = parse_str.find(' ')
#         print(space_pos, parse_str)
#         label = parse_str[1:space_pos]
#         print(label)

#     for child in spacy_tree._.children:
#         show_spacy_const_tree(child, level=level+1)

# show_spacy_const_tree(sent)