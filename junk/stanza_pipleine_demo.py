import stanza

#stanza.download('en') # Only need to run this once.

processors = 'tokenize,pos,lemma,depparse,ner, constituency'

nlp = stanza.Pipeline('en', processors=processors, use_gpu=True, pos_batch_size=3000) # Build the pipeline, specify part-of-speech processor's batch size


text = "Barack Obama was not born in Hawaii."


doc = nlp(text) # Run the pipeline on the input text

#print(doc) # Look at the result

for sentence in doc.sentences:
    print(sentence.ents)
    print(sentence.dependencies)
    print(sentence.ents)
    print(sentence.constituency)