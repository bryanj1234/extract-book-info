from lib_corenlp import *
from my_conparse_tree import MyConParseTree
from lib_GUM import *
import tempfile

text_list = [
   # "We can generally distinguish between classiﬁcation problems , in which the label t is discrete , and regression problems , in which variable t is continuous .",
   #  "It’s important to think about why we make certain deserts, and why we choose to eat food.",
   #  "This is the taxi which is yellow",
   #  "This is the taxi that is yellow",
   #  "This is the store where we buy stuff",
   #  "This is the store in which we buy stuff",
   #  "i want to make, and choose, and run to the store, but always fly.",
   #  "You want and I love",
   #  "You and I want and love",
   #  "The red-blooded pirate, wimpy accountant, or their friends love dogs",
   #  "he throws the red, blue, and yellow base car balls into the woods",
   #  "he throws the red, blue, and yellow base car balls into the woods, but if it's rainy he will sleep",
   #  "he throws the red, blue and yellow base balls and bats",
   #  "he quickly, efficiently, and accurately throws the balls.",
   #  "the red, blue, and yellow balls are great.",
   #  "he eats the pizza, pie, and ice cream.",
   #  "he ran, jumped, and swam, quickly and efforlessly",
   #  "Sally and Bob eat and drink the pizza and the soda.",
   #  "Sally and Bob each eat the hot pizza and the coldest soda.",
   #  "On Tuesday, Sally and the extremely fast runner and batter, throw jim the ball, which is red.",

   #  "Sally runs and jumps",
   #  "Sally throws the soccer ball",
   #  "Sally throws soccer balls and foot balls",
   #  "The soccer ball was thrown by Sally.",
   #  "Sally throws the dog the ball, which is red.",
   #  "Sally throws the dog the ball.",
   #  "Sally throws the blue ball, which usually is red.",
   #  "One dog runs faster?",
   #  "Bryan asked Nancy, who was pleased.",
   #  "Bryan asked Nancy, whose cat was red.",
   #  "The ball is red",
   #  "red is the ball",
   #  "red is the ball which bounces"
   #  "the dog will be the cat",
   #  "I throw the ball",
   #  "I throw to Bryan's friend of a friend"
   #  "This extremely and quietly greenish and red or blue dog quite quickly but very very briskly runs .",
   #  "I throw to him",
   #  "I give to the charity from Lansing.",
   #  "I usually run fast",

   #  "Sally throws the ball into the woods",
   #  "the dog from venus and the cat from Mars eat food",
   #  "I talk to the dog from Mars",
   #  "in the beginning, I usually practice",

   #  "the red dog and the brown horse ran",
   #  "I quickly and efficiently followed Nancy, James, and Blair"
   #  "Sally runs and swims",
   #  "Sally has red and brown potatos",
   #  "Sally runs very quickly and quite efficiently",
   #  "we had his and her towels",
   #  "He ran over the fields and into the woods"
   #  "The dog from the pound and from the store likes the kibble",
   #  "Sally is very mad and quite angry",
   #  "I should hit the ball",
   #  "Bryan quite quickly and very very briskly runs .",

   #  "the dog from and in the shelter runs quickly and flies fast",
   #  "the dog runs fast from the shelter",
   #  "the dog runs from the shelter fast",
   #  "the dog runs and flies from the shelter",
   #  "from the shelter, the dog runs and flies",
   #  "the dog runs from the for storms shelter"      # Good example of NML contituent!

   #  "the dog quickly runs through the woods",
   #  "the red and blue dog briskly runs joyfully through the woods quickly",
   #  "I like the red-running and blue or green dog or cat.",
   #  "I like the previously living red green dog, which lives happily joyfully and sadly lethargically."

   #  "to the store runs the dog",
   #  "the dog runs to the store",
   #  "quickly runs the dog",

   #  "the ball was hit by the boy",

   #  "we throw or run and hit the ball and wooden bat",
   #  "we quickly throw, and nimbly catch, the balls",
   #  "we throw the ball",
   #  "we hit the red and blue balls",
   #  "over the river and through the woods, to grandmother's house we go."

   #  "usually red is the ball",
   #  "the dog is usually brown",
   #  "They wonder which wildebeest the lions will devour",
   #  "I chase the dog which runs",
   #  "Sally throws the ball, which is red.",

   #"Claiming that there is a relation, however, is empirically dubious, according to Street."
   #  "Claiming that there is a relation means that nothing works."
   #  "The dog is the cat"
   #  "The man was killed by the dog"
   #  "the brown and black dog from Detroit quickly runs and jumps into the leafy woods"

   #  "the red blue green dog is fun"

   #  "Each time the Start button is pressed and a new classifier is built and evaluated, a new entry appears in the Result List panel in the lower left corner of Figure.",
   #  "In Chapter 6 we see that we can improve upon least squares using ridge regression, the lasso, principal components regression, and other techniques.",
   #  "We do this by examining very simple extensions of linear models like polynomial regression and step functions, as well as more sophisticated approaches such as splines, local regression, and generalized additive models.",
   #  "Regression splines are more ﬂexible than polynomials and step functions, and in fact are an extension of the two.",
   #  "The regions are allowed to overlap, and indeed they do so in a very smooth way.",
   #  "In ridge regression, each least squares coeﬃcient estimate is shrunken by the same proportion.",
   #  "In contrast, the lasso shrinks each least squares coeﬃcient towards zero by a constant amount, λ/2.",
   #  "The least squares coeﬃcients that are less than λ/2 in absolute value are shrunken entirely to zero.",
   #  "We will now show that one can view ridge regression and the lasso through a Bayesian lens.",

   #  "The red dog likes the really fast wild cat.",

    # "Dogs run.",
    # "Cats are also great.",
    # "I like the dog and the cat which runs.",
    # "We were asked by George to leave, which means that we weren't wanted here.",
    # "Fruit flies like a banana.",
    #"There is a red car. The red car is blue.",
    #"The dog has been selectively bred over millennia for various behaviors",
    #"The cat flies.",
    # "The Red dogs run. The red dog flies.",
    #"The dog or domestic dog (Canis familiaris or Canis lupus familiaris) is a domesticated descendant of the wolf which is characterized by an upturning tail. The dog derived from an ancient, extinct wolf, and the modern grey wolf is the dog's nearest living relative. The dog was the first species to be domesticated, by hunter–gatherers over 15,000 years ago, before the development of agriculture. Due to their long association with humans, dogs have expanded to a large number of domestic individuals and gained the ability to thrive on a starch-rich diet that would be inadequate for other canids.",
    #"Over the millennia, dogs became uniquely adapted to human behavior, and the human-canine bond has been a topic of frequent study. The dog has been selectively bred over millennia for various behaviors, sensory capabilities, and physical attributes. Dog breeds vary widely in shape, size, and color.",
    #"Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much. Spain is great."

    "Such a set of rules can be written as a logic expression in what is called disjunctive normal form: that is, as a disjunction (OR) of conjunctive (AND) conditions."
]

# # Get random GUM sentence for deprel.
# idx_sentences_by_deprel = get_GUM_dependency_info()
# deprel = 'xcomp' # acl   advcl   appos   csubj   ccomp   xcomp
# rec = get_random_sentence_record_for_deprel(deprel, idx_sentences_by_deprel)
# print(rec)
# text_list = [rec[4]]

#nlp_package_str = 'STANZA_WITH_CORENLP'
nlp_package_str = 'SPACY'
nlp_loader = NLPClientLoader(nlp_package_str, reload_every=100)

graph_num = 0

for text in text_list:
    print(text)

    #-----------------------------------------------------------------------------------------

    nlp_client = nlp_loader.load_nlp_client()

    sentence_trees = get_sentence_trees(text, nlp_client, nlp_package_str)


    #-----------------------------------------------------------------------------------------

    idx_nodes = {}
    idx_edges = {}
    idx_word_UIDs = {}
    idx_main_UIDs = {}
    idx_copula_edges = {}
    idx_target_edges = {}
    idx_orig_order_edges = {}
    idx_WH_word_edges = {}

    sentence_num = 0
    for sentence_tree in sentence_trees:

        parse_tree = sentence_tree['parse_tree']
        idx_words = sentence_tree['idx_words']

        # Massage the constituency tree a little, and add the node UIDs.
        sentence_num += 1
        node_prefix = "sent_" + str(sentence_num)
        dep_tree, idx_dep_tree_words = parse_tree.preprocess_for_graph(node_prefix=node_prefix)

        #------------------------------------------------------------------------

        # parse_tree.pprint(all_info=False)
        # core_nlp_print_dependency_tree(dep_tree)
        # print(sentence_tree['text'])

        #print("#######################################################################")

        new_idx_nodes, new_idx_edges, new_idx_word_UIDs, new_idx_main_UIDs, new_idx_copula_edges, \
                new_idx_target_edges, new_idx_orig_order_edges, new_idx_WH_word_edges \
            = get_graph_from_constituency_tree(parse_tree, idx_dep_tree_words)

        # Merge existing and new information
        idx_nodes = {**idx_nodes, **new_idx_nodes}
        idx_edges = {**idx_edges, **new_idx_edges}
        idx_word_UIDs = {**idx_word_UIDs, **new_idx_word_UIDs}
        idx_main_UIDs = {**idx_main_UIDs, **new_idx_main_UIDs}
        idx_copula_edges = {**idx_copula_edges, **new_idx_copula_edges}
        idx_target_edges = {**idx_target_edges, **new_idx_target_edges}
        idx_orig_order_edges = {**idx_orig_order_edges, **new_idx_orig_order_edges}
        idx_WH_word_edges = {**idx_WH_word_edges, **new_idx_WH_word_edges}

    # Add edges between noun nodes that have the same word.
    idx_identical_noun_edges = {}
    add_edges_between_identical_nouns(idx_nodes, idx_edges, idx_identical_noun_edges)

    # Add edges for coreferences
    idx_coreference_edges = {}
    add_coreference_edges(idx_nodes, idx_edges, idx_coreference_edges)

    graph_num += 1
    render_graph_graphviz(graph_num, idx_nodes, idx_edges, idx_word_UIDs, idx_main_UIDs, idx_copula_edges, idx_target_edges,
                          idx_orig_order_edges, idx_WH_word_edges, idx_identical_noun_edges, idx_coreference_edges)

