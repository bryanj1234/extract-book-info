import sys
import os
import io
import pathlib
import re
import random

def get_GUM_dependency_info():
    gum_dir_path_str = '/home/bryan/Documents/DEV/not-version-controlled/GUM - Georgetown University Multilayer Corpus'
    dependencies_dir = os.path.join(gum_dir_path_str, 'dep')

    # Get names of dependency files
    dep_file_name_str_recs = []
    with os.scandir(dependencies_dir) as it:
        for entry in it:
            if entry.is_file():
                dep_file_name_str_recs.append(entry.name)
    dep_file_name_str_recs = sorted(dep_file_name_str_recs)


    idx_sentences_by_deprel = {}

    cur_sentence_text = False
    for file_str in dep_file_name_str_recs:
        file_name_path_str = os.path.join(dependencies_dir, file_str)
        #print(file_name_path_str)
        with open(file_name_path_str, 'r') as rfh:
            lines = rfh.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('# text = '):
                    cur_sentence_text = line.replace('# text = ', '')
                    #print(cur_sentence_text)
                else:
                    line_parts = line.split('\t')
                    if re.match('[0-9]+', line_parts[0]):
                        if line_parts[0].isnumeric():
                            word_id = int(line_parts[0])
                            word = line_parts[1]
                            pos = line_parts[4]
                            deprel = line_parts[7]
                            #print(word_id, word, pos, deprel)
                            if not deprel in idx_sentences_by_deprel:
                                idx_sentences_by_deprel[deprel] = []
                            idx_sentences_by_deprel[deprel].append([word, word_id, pos, deprel, cur_sentence_text])

    return idx_sentences_by_deprel


def get_random_sentence_record_for_deprel(deprel, idx_sentences_by_deprel):
    if deprel in idx_sentences_by_deprel:
        return idx_sentences_by_deprel[deprel][random.randrange(len(idx_sentences_by_deprel[deprel]))]
    else:
        return False







