# https://stanfordnlp.github.io/stanza/
# pip install stanza
# Import stanza
import stanza

# Import client module
from stanza.server import CoreNLPClient

# example text
print('---')
print('input text')
print('')

#text = "Fruit flies like a banana."
text = "We do this by examining very simple extensions of linear models like polynomial regression and step functions, as well as more sophisticated approaches such as splines, local regression, and generalized additive models."

print(text)


# client = CoreNLPClient(
#     annotators=['tokenize','ssplit','pos','lemma','ner','parse','depparse','coref'],
#     memory='4G',
#     endpoint='http://localhost:9001',
#     be_quiet=True)
# print(client)

client = CoreNLPClient(start_server=stanza.server.StartServer.DONT_START, endpoint='http://localhost:9000')


# submit the request to the server
ann = client.annotate(text)

# get the first sentence
sentence = ann.sentence[0]

# get the dependency parse of the first sentence
print('---')
print('dependency parse of first sentence')
dependency_parse = sentence.basicDependencies
print(dependency_parse)

# get the constituency parse of the first sentence
print('---')
print('constituency parse of first sentence')
constituency_parse = sentence.parseTree
print(constituency_parse)

# get the first subtree of the constituency parse
print('---')
print('first subtree of constituency parse')
print(constituency_parse.child[0])

# get the value of the first subtree
print('---')
print('value of first subtree of constituency parse')
print(constituency_parse.child[0].value)

# get the first token of the first sentence
print('---')
print('first token of first sentence')
token = sentence.token[0]
print(token)

# get the part-of-speech tag
print('---')
print('part of speech tag of token')
token.pos
print(token.pos)

# get the named entity tag
print('---')
print('named entity tag of token')
print(token.ner)

# get an entity mention from the first sentence
print('---')
print('first entity mention in sentence')
print(sentence.mentions[0])

# access the coref chain
print('---')
print('coref chains for the example')
print(ann.corefChain)

# Use tokensregex patterns to find who wrote a sentence.
pattern = '([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/'
matches = client.tokensregex(text, pattern)
# sentences contains a list with matches for each sentence.
assert len(matches["sentences"]) == 3
# length tells you whether or not there are any matches in this
assert matches["sentences"][1]["length"] == 1
# You can access matches like most regex groups.
matches["sentences"][1]["0"]["text"] == "Chris wrote a simple sentence"
matches["sentences"][1]["0"]["1"]["text"] == "Chris"

# Use semgrex patterns to directly find who wrote what.
pattern = '{word:wrote} >nsubj {}=subject >obj {}=object'
matches = client.semgrex(text, pattern)
# sentences contains a list with matches for each sentence.
assert len(matches["sentences"]) == 3
# length tells you whether or not there are any matches in this
assert matches["sentences"][1]["length"] == 1
# You can access matches like most regex groups.
matches["sentences"][1]["0"]["text"] == "wrote"
matches["sentences"][1]["0"]["$subject"]["text"] == "Chris"
matches["sentences"][1]["0"]["$object"]["text"] == "sentence"
