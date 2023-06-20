import lib_corenlp
from stanza.models.constituency.parse_tree import Tree
from collections import deque

## NOTES #######################################################
# See lib_CoreNLP.py for list of dependency relations used

# class Tree(StanzaObject):
#     def __init__(self, label=None, children=None):
#     def is_leaf(self):
#     def is_preterminal(self):
#     def yield_reversed_preterminals(self):
#     def leaf_labels(self):
#     def preterminals(self):
#     def __repr__(self):
#     def __eq__(self, other):
#     def depth(self):
#     def visit_preorder(self, internal=None, preterminal=None, leaf=None):
#     def get_compound_constituents(trees):
#     def simplify_labels(self, pattern=CONSTITUENT_SPLIT):
#     def remap_constituent_labels(self, label_map):
#     def remap_words(self, word_map):
#     def replace_words(self, words):
#     def prune_none(self):
#     @staticmethod
#     def get_unique_constituent_labels(trees):
#     @staticmethod
#     def get_unique_tags(trees):
#     @staticmethod
#     def get_unique_words(trees):
#     @staticmethod
#     def get_rare_words(trees, threshold=0.05):
#     @staticmethod
#     def get_root_labels(trees):
#     @staticmethod

# Warning! Don't add or remove nodes to MyConParseTree.children, otherwise child.parent will not be correct.
class MyConParseTree(Tree):
    def __init__(self, label=None, children=None):
        self.label = label
        self.children = children if children else []
        # Make sure to point child trees back up to parent!
        for child_tree in self.children:
            child_tree.parent = self
        self.orig_idx_words = None    # Added from the original sentence before running preprocess_for_graph()
        self.parent = None
        self.word_info = None
        self.idx_word_ids = None
        self.idx_unresolved_head_ids = None
        self.idx_deprel_by_unresolved_head_ids = None
        self.UID = None
        self.subsentence_UID = None
        self.idx_UIDs_to_subtrees = None
        self.subtree_num = None
        self.level = None
        self.graph_pos = None

    def add_child(self, child_tree):
        self.children.append(child_tree)
        child_tree.parent = self

    def add_child_at_beginning(self, child_tree):
        self.children.insert(0, child_tree)
        child_tree.parent = self

    def add_child_at_position(self, ind, child_tree):
        self.children.insert(ind, child_tree)
        child_tree.parent = self

    def add_children(self, children):
        for child_tree in children:
            self.add_child(child_tree)

    def remove_children(self):
        for child in self.children:
            child.parent = None
        self.children = []

    # Use this to make sure all the child/parent links in the tree are correct.
    # Recursive.
    def is_valid(self):
        bool_valid = True
        for child_tree in self.children:
            if child_tree.parent is not self:
                bool_valid = False
            else:
                bool_valid = child_tree.is_valid()
            if not bool_valid:
                break
        assert bool_valid, 'Invalid tree:' + str(self)
        return bool_valid

    # deque() version.
    @staticmethod
    def get_from_core_nlp_contree(core_nlp_contree):
        nodes = deque()

        root_my_tree = False

        nodes.append((False, core_nlp_contree))
        while len(nodes) > 0:
            node = nodes.pop()

            parent_tree = node[0]
            nlp_contree = node[1]

            new_my_con_tree = MyConParseTree(label=nlp_contree.value)
            if parent_tree:
                parent_tree.add_child(new_my_con_tree)
            else:
                root_my_tree = new_my_con_tree

            for child_CoreNLP_contree in reversed(nlp_contree.child):
                nodes.append((new_my_con_tree, child_CoreNLP_contree))

        return root_my_tree

    # deque() version.
    @staticmethod
    def get_from_stanza_contree(core_nlp_contree):
        nodes = deque()

        root_my_tree = False

        nodes.append((False, core_nlp_contree))
        while len(nodes) > 0:
            node = nodes.pop()

            parent_tree = node[0]
            nlp_contree = node[1]

            new_my_con_tree = MyConParseTree(label=nlp_contree.label)
            if parent_tree:
                parent_tree.add_child(new_my_con_tree)
            else:
                root_my_tree = new_my_con_tree

            for child_CoreNLP_contree in reversed(nlp_contree.children):
                nodes.append((new_my_con_tree, child_CoreNLP_contree))

        return root_my_tree

    # deque() version.
    @staticmethod
    def get_from_spacy_contree(idx_words, spacy_tree):
        nodes = deque()

        root_my_tree = MyConParseTree(label='ROOT')

        nodes.append((root_my_tree, spacy_tree))
        while len(nodes) > 0:
            node = nodes.pop()

            parent_tree = node[0]
            nlp_contree = node[1]

            label = "_NONE_"
            token_index = False
            if len(nlp_contree._.labels) > 0:
                label = nlp_contree._.labels[0]
            else:
                token_index = 1 + nlp_contree.start - spacy_tree.start
                label = idx_words[token_index]['pos']
            new_my_con_tree = MyConParseTree(label=label)
            parent_tree.add_child(new_my_con_tree)

            if len(list(nlp_contree._.children)) > 0:
                for child_CoreNLP_contree in reversed(list(nlp_contree._.children)):
                    nodes.append((new_my_con_tree, child_CoreNLP_contree))
            else:
                # As far as I can tell, there are two scenarios, depending on whether the token has siblings:
                #   1) (VBZ runs)           Has siblings, len(nlp_contree._.labels) == 0.
                #   2) (VP (VBZ runs))      No siblings, len(nlp_contree._.labels) > 0.
                if len(nlp_contree._.labels) == 0:
                    leaf_tree = MyConParseTree(label=nlp_contree.text)
                    new_my_con_tree.add_child(leaf_tree)
                else:
                    # Need to inject an extra preterminal node.
                    preterminal_tree = MyConParseTree(label=nlp_contree[0].tag_)
                    leaf_tree = MyConParseTree(label=nlp_contree.text)
                    preterminal_tree.add_child(leaf_tree)
                    new_my_con_tree.add_child(preterminal_tree)

        return root_my_tree


    def add_idx_words(self, idx_words):
        self.orig_idx_words = idx_words

        # Add idx_words info to preterminals, and delete any preterminals where the word_id is no longer in idx_words.
        preterminals = self.preterminals()
        for ii in range(len(preterminals)):
            pt_tree = preterminals[ii]
            word_id = ii + 1
            word = idx_words[word_id]
            pt_tree.word_info = {
                'id': word['id'], 'text': word['text'], 'head': word['head'], 'pos': word['pos'],
                'deprel': word['deprel'],
                'lemma': word['lemma'],
                'doc_token_index':word['doc_token_index'] if 'doc_token_index' in word else False,
                'coref_res_doc_token_index':word['coref_res_doc_token_index'] if 'coref_res_doc_token_index' in word else False,
                'all_coref_res_doc_token_index':word['all_coref_res_doc_token_index'] if 'all_coref_res_doc_token_index' in word else False,
                'coref_resolved_text':word['coref_resolved_text'] if 'coref_resolved_text' in word else False,
                'coref_resolved_lemma':word['coref_resolved_lemma'] if 'coref_resolved_lemma' in word else False,
            }

    # Recursively populates these:
    #     self.idx_word_ids = None
    #     self.idx_unresolved_head_ids = None
    #     self.idx_deprel_by_unresolved_head_ids = None
    def populate_master_info_first_pass(self, idx_words):

        # Make sure I did things in the right order.
        if self.is_preterminal() and self.word_info is None:
            assert "Bryan made a bug: is_preterminal() vs word_info"

        self.idx_word_ids = {}
        self.idx_unresolved_head_ids = {}
        self.idx_deprel_by_unresolved_head_ids = {}

        # We've reached a preterminal.
        #   ("word_info" was added by some previous call to add_idx_words().)
        if self.is_preterminal():
            self.idx_word_ids[self.word_info['id']] = True
            self.idx_unresolved_head_ids = {self.word_info['head']: True}
            self.idx_deprel_by_unresolved_head_ids[self.word_info['head']] = {}
            self.idx_deprel_by_unresolved_head_ids[self.word_info['head']][self.word_info['deprel']] = [
                self.word_info['text'], self.word_info['id']]

        # We're closer to the root than a preterminal.
        else:
            idx_child_idx_unresolved_head_ids = {}
            idx_child_idx_deprel_by_unresolved_head_ids = {}
            for child_tree in self.children:
                child_tree.populate_master_info_first_pass(idx_words)

                # Things to always roll up.
                # idx_word_ids
                for word_id in child_tree.idx_word_ids:
                    self.idx_word_ids[word_id] = True
                for head_id in child_tree.idx_unresolved_head_ids:
                    idx_child_idx_unresolved_head_ids[head_id] = True
                # idx_deprel_by_unresolved_head_ids
                for head_id in child_tree.idx_deprel_by_unresolved_head_ids:
                    if not head_id in idx_child_idx_deprel_by_unresolved_head_ids:
                        idx_child_idx_deprel_by_unresolved_head_ids[head_id] = {}
                    for deprel in child_tree.idx_deprel_by_unresolved_head_ids[head_id]:
                        if not deprel in idx_child_idx_deprel_by_unresolved_head_ids[head_id]:
                            idx_child_idx_deprel_by_unresolved_head_ids[head_id][deprel] = []
                        idx_child_idx_deprel_by_unresolved_head_ids[head_id][deprel].extend(
                            child_tree.idx_deprel_by_unresolved_head_ids[head_id][deprel])

            # Figure out which words are still not resolved at this point.
            for head_id in idx_child_idx_unresolved_head_ids:
                if not head_id in self.idx_word_ids:
                    self.idx_unresolved_head_ids[head_id] = True

            # Add into self.idx_deprel_by_unresolved_head_ids for head_ids which are still unresolved
            for head_id in idx_child_idx_deprel_by_unresolved_head_ids:
                if head_id in self.idx_unresolved_head_ids:
                    if not head_id in self.idx_deprel_by_unresolved_head_ids:
                        self.idx_deprel_by_unresolved_head_ids[head_id] = {}
                    for deprel in idx_child_idx_deprel_by_unresolved_head_ids[head_id]:
                        if not deprel in self.idx_deprel_by_unresolved_head_ids[head_id]:
                            self.idx_deprel_by_unresolved_head_ids[head_id][deprel] = []
                        self.idx_deprel_by_unresolved_head_ids[head_id][deprel].extend(
                            idx_child_idx_deprel_by_unresolved_head_ids[head_id][deprel])

    # Recursively populates these:
    #     self.UID = None
    #     self.subsentence_UID = None
    #     self.idx_ui_ds_to_subtrees = None
    #     self.subtree_num = None
    #     self.level = None
    def populate_master_info_second_pass(self, node_prefix, max_id=0, level=0):
        max_id += 1

        # UID related stuff -------------------------------------------
        if ':' in node_prefix:
            assert False, "node_prefix cannot have ':' (colons), because it will break GraphViz."
        UID = node_prefix + '_' + str(max_id)
        self.UID = UID
        self.level = level
        self.subtree_num = max_id

        # Only succeeds for the tree root,
        #   because we're setting the value for the child before recursing.
        if self.idx_UIDs_to_subtrees is None:
            self.idx_UIDs_to_subtrees = {}
        self.idx_UIDs_to_subtrees[UID] = self

        # Deal with idx_ui_ds_to_subtrees
        if not self.parent or self.label in ['S', 'SINV', 'SBAR', 'SBARQ', 'SQ']:
            self.subsentence_UID = self.UID
        else:
            # Get from parent.
            self.subsentence_UID = self.parent.subsentence_UID

        for child_tree in self.children:
            child_tree.idx_UIDs_to_subtrees = self.idx_UIDs_to_subtrees
            max_id = child_tree.populate_master_info_second_pass(node_prefix, max_id=max_id, level=level + 1)

        return max_id

    def get_idx_deprel_by_unresolved_head_ids_highest(self, idx_deprel_by_unresolved_head_ids_highest=None):
        if idx_deprel_by_unresolved_head_ids_highest is None:
            idx_deprel_by_unresolved_head_ids_highest = {}

        for head_id in self.idx_deprel_by_unresolved_head_ids:
            for deprel in self.idx_deprel_by_unresolved_head_ids[head_id]:
                idx = str(head_id) + ':' + deprel
                if not idx in idx_deprel_by_unresolved_head_ids_highest:
                    idx_deprel_by_unresolved_head_ids_highest[idx] \
                        = [head_id, deprel, self.label, self.level, self.UID,
                           self.idx_deprel_by_unresolved_head_ids[head_id][deprel]]

        # Recurse
        if not self.is_preterminal():
            for child_tree in self.children:
                child_tree.get_idx_deprel_by_unresolved_head_ids_highest(
                    idx_deprel_by_unresolved_head_ids_highest=idx_deprel_by_unresolved_head_ids_highest)

        return idx_deprel_by_unresolved_head_ids_highest

    def print_idx_deprel_by_unresolved_head_ids_highest(self, idx_words):
        idx_deprel_by_unresolved_head_ids_highest = self.get_idx_deprel_by_unresolved_head_ids_highest()
        for idx in idx_deprel_by_unresolved_head_ids_highest:
            head_id = idx_deprel_by_unresolved_head_ids_highest[idx][0]
            if head_id == 0:
                continue
            print(head_id, idx_deprel_by_unresolved_head_ids_highest[idx])
            print("\thead:", idx_words[head_id]['text'])

    def pprint(self, all_info=False, level=0):
        if all_info:
            print("\t" * level, self.label,
                  # self.UID,
                  "\t ->" + self.parent.label if self.parent else '')
            print("\t" * level, "    word_info:", self.word_info)
            print("\t" * level, "    idx_word_ids:", self.idx_word_ids)
            print("\t" * level, "    idx_unresolved_head_ids:", self.idx_unresolved_head_ids)
            print("\t" * level, "    idx_deprel_by_unresolved_head_ids:", self.idx_deprel_by_unresolved_head_ids)
        else:
            print("\t" * level, self.label, self.UID, self.subsentence_UID)
        if not (all_info and self.is_preterminal()):
            for child_tree in self.children:
                child_tree.pprint(all_info=all_info, level=level + 1)

    def get_node_list(self, node_list=None):
        if node_list is None:
            node_list = []
        node_list.append(self)
        # Recurse
        for child_tree in self.children:
            child_tree.get_node_list(node_list=node_list)

        return node_list

    def get_text(self):
        return ' '.join(pt_tree.children[0].label for pt_tree in self.preterminals())


    ### Methods to normalize the tree #########################################################################

    def remove_punctuation(self):
        preterminals = self.preterminals()
        for pt_node in preterminals:
            if pt_node.word_info and pt_node.word_info['deprel'] == 'punct':
                if len(pt_node.parent.children) > 1:
                    pt_node.parent.children.remove(pt_node)


    def get_dependency_tree(self):
        idx_preterms_by_word_id = {}
        for pt_tree in self.preterminals():
            idx_preterms_by_word_id[pt_tree.word_info['id']] = pt_tree

        # Initialize children.
        idx_dep_tree = {}
        for word_id in idx_preterms_by_word_id:
            word = idx_preterms_by_word_id[word_id].word_info
            word['children'] = []
            idx_dep_tree[word_id] = word

        dep_tree = False
        for word_id in idx_dep_tree:
            word = idx_dep_tree[word_id]
            head_id = word['head']
            if head_id != 0:
                head_dep_tree_node = idx_dep_tree[head_id]
                head_dep_tree_node['children'].append(word)
            else:  # Found the root.
                dep_tree = word

        return dep_tree, idx_dep_tree

    # Preprocess the parse tree a little.
    # Flatten stuff and make compund nouns, verbs, etc...
    def preprocess_for_graph(self, node_prefix='my_tree'):

        # Make sure everything still looks OK.
        assert self.is_valid(), "Ivalid parse tree!"

        # Populate master info
        self.populate_master_info_first_pass(self.orig_idx_words)
        self.populate_master_info_second_pass(node_prefix=node_prefix)

        dep_tree, idx_dep_tree_words = self.get_dependency_tree()

        return dep_tree, idx_dep_tree_words
