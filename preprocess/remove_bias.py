"""
Step 4
This file finds the valid GO terms that satisfy rules:
1. the go term has at least 30 proteins annotated with it
and
2. none of the children of the go term satisfies rule 1
and modifies the files: HumanPPI_GO_BP.txt, HumanPPI_GO_MF.txt, HumanPPI_GO_CC.txt for both 700 and 900 ppi types
"""
import time
import numpy as np
import pandas as pd


def get_children_relations(file):
    """ Finds children relations for each go term

    :param file: file path to file that contains the go relations
    :type file: str
    :return: parent children relations in dictionary
    :rtype: dict
    """
    go_bp = pd.read_csv(file, header=None, sep=' ', names=['term1', 'relation', 'term2'])
    # filter for child-parent relations
    go_bp = go_bp.loc[go_bp['relation'] == 'is_a']
    parents = set(go_bp['term2'].values.tolist())
    relations = dict()
    for term in parents:
        relations[term] = set(go_bp.loc[go_bp['term2'] == term]['term1'].values.tolist())
    return relations


def remove_bias(annotations_file, go_file, output_file):
    """ Finds the valid GO terms from annotations file that satisfy rules:
        1. the go term has at least 30 proteins annotated with it
        and
        2. none of the children of the go term satisfies rule 1
    and saves them in output file

    :param annotations_file: file path to protein-go annotations
    :type annotations_file: str
    :param go_file: file path to file that contains the go relations
    :type go_file: str
    :param output_file: file path to save the filtered annotations
    :return: None
    """
    counts = pd.read_csv(annotations_file, sep='\t', header=0).groupby(['go_id']).count()
    go_terms = set(counts.index.values.tolist())
    children_relations = get_children_relations(go_file)
    bias_terms = set()
    for go_term in go_terms:
        num_parent = counts.get_value(go_term, 'protein_id')
        if num_parent < 30:
            bias_terms.add(go_term)
        else:
            if go_term not in children_relations:
                continue
            for child in children_relations[go_term]:
                if child not in go_terms:
                    continue
                num_child = counts.get_value(child, 'protein_id')
                if num_child >= 30:
                    bias_terms.add(go_term)
                    break

    keep = list(go_terms - bias_terms)
    annotations = pd.read_csv(annotations_file, sep='\t', header=0)
    annotations = annotations.join(pd.DataFrame({'go_id': keep}).set_index('go_id'), how='inner', on='go_id')
    annotations.to_csv(output_file, sep='\t', index=False)


if __name__ == '__main__':
    print("(" + time.strftime("%c") + ")  Remove bias in the HumanPPI700 GO BP file...")
    remove_bias('../data/human_ppi_700/HumanPPI_GO_BP.txt',
                '../data/go/BPGOfull.txt',
                '../data/human_ppi_700/HumanPPI_GO_BP_no_bias.txt')
    print("(" + time.strftime("%c") + ")  Remove bias in the HumanPPI700 GO MF file...")
    remove_bias('../data/human_ppi_700/HumanPPI_GO_MF.txt',
                '../data/go/MFGOfull.txt',
                '../data/human_ppi_700/HumanPPI_GO_MF_no_bias.txt')
    print("(" + time.strftime("%c") + ")  Remove bias in the HumanPPI700 GO CC file...")
    remove_bias('../data/human_ppi_700/HumanPPI_GO_CC.txt',
                '../data/go/CCGOfull.txt',
                '../data/human_ppi_700/HumanPPI_GO_CC_no_bias.txt')

    print("(" + time.strftime("%c") + ")  Remove bias in the HumanPPI900 GO BP file...")
    remove_bias('../data/human_ppi_900/HumanPPI_GO_BP.txt',
                '../data/go/BPGOfull.txt',
                '../data/human_ppi_900/HumanPPI_GO_BP_no_bias.txt')
    print("(" + time.strftime("%c") + ")  Remove bias in the HumanPPI900 GO MF file...")
    remove_bias('../data/human_ppi_900/HumanPPI_GO_MF.txt',
                '../data/go/MFGOfull.txt',
                '../data/human_ppi_900/HumanPPI_GO_MF_no_bias.txt')
    print("(" + time.strftime("%c") + ")  Remove bias in the HumanPPI900 GO CC file...")
    remove_bias('../data/human_ppi_900/HumanPPI_GO_CC.txt',
                '../data/go/CCGOfull.txt',
                '../data/human_ppi_900/HumanPPI_GO_CC_no_bias.txt')
