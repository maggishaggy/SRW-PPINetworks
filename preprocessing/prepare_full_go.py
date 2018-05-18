"""
Step 2
This file connects string-db proteins with GO annotations according to GO Consortium
"""
import time
import pandas as pd


def read_string_aliases(reverse=True):
    """ Reads string-db protein aliases

    :param reverse: flag, if True the result is dictionary with key alias and value string protein id,
                    else the key is string protein id and the value is the alias
    :type reverse: bool
    :return: aliases in dictionary
    :rtype: dict
    """
    aliases = dict()
    with open('../data/9606.protein.aliases.v10.5.txt', 'r') as file:
        file.readline()
        for line in file:
            line.strip()
            parts = line.split("\t")
            if reverse:
                aliases[parts[1]] = parts[0]
            else:
                aliases[parts[0]] = parts[1]
    return aliases


def get_go_terms_relationships():
    """ Reads all go term relationships from BP, MF and CC term files

    :return: go terms relationships
    :rtype: pandas.DataFrame
    """
    print("(" + time.strftime("%c") + ")  Filling child-parent relationship dictionary...")

    go_bp = pd.read_csv('../data/go/BPGOfull.txt', header=None, sep=' ', names=['term1', 'relation', 'term2'])
    # filter for child-parent relations
    go_bp = go_bp.loc[((go_bp['relation'] == 'is_a') | (go_bp['relation'] == 'part_of'))]

    go_mf = pd.read_csv('../data/go/MFGOfull.txt', header=None, sep=' ', names=['term1', 'relation', 'term2'])
    # filter for child-parent relations
    go_mf = go_mf.loc[((go_mf['relation'] == 'is_a') | (go_mf['relation'] == 'part_of'))]

    go_cc = pd.read_csv('../data/go/CCGOfull.txt', header=None, sep=' ', names=['term1', 'relation', 'term2'])
    # filter for child-parent relations
    go_cc = go_cc.loc[((go_cc['relation'] == 'is_a') | (go_cc['relation'] == 'part_of'))]

    return pd.concat([go_bp, go_mf, go_cc], ignore_index=True)


def get_go_terms():
    """ Reads all go terms from BP, MF and CC term files

    :return: go terms
    :rtype: list
    """
    go_bp = pd.read_csv('../data/go/BPGOterms.txt', header=None, sep=' ', names=['term'])
    go_mf = pd.read_csv('../data/go/MFGOterms.txt', header=None, sep=' ', names=['term'])
    go_cc = pd.read_csv('../data/go/CCGOterms.txt', header=None, sep=' ', names=['term'])

    return go_bp['term'].values.tolist() + go_mf['term'].values.tolist() + go_cc['term'].values.tolist()


def get_all_go_terms(go_id, go_relations):
    """ Finds all parent terms from the GO graph (tree) for given go term

    :param go_id: id of the go term
    :type go_id: str
    :param go_relations: parent child go relations
    :type go_relations: pandas.DataFrame
    :return: all go terms in the
    :rtype: set
    """
    all_go_annotations = set()
    child_go_terms = go_relations['term1'].values.tolist()
    parents_go = set()
    parents_go.add(go_id)

    while len(parents_go) != 0:
        go_id = parents_go.pop()
        all_go_annotations.add(go_id)
        if go_id in child_go_terms:
            parents = go_relations.loc[go_relations['term1'] == go_id]['term2'].values.tolist()
            parents_go.update(parents)

    return all_go_annotations


def get_protein_go_annotations(include_parents=True):
    """ Annotates proteins from string-db with go terms

    :param include_parents: flag, if True include parents of term in annotations
    :type include_parents: bool
    :return: dictionary with protein as key and set of go terms as values
    :rtype: dict
    """
    string_aliases = read_string_aliases()
    names = ['db', 'db_object_id', 'db_object_symbol', 'qualifier', 'go_id',
             'db_reference', 'evidence_code', 'with_from', 'aspect', 'db_object_name',
             'db_object_synonym', 'db_object_type', 'taxon', 'date', 'assigned_by',
             'annotation_extension', 'gene_product_form_id']
    goa_human = pd.read_csv('../data/go/goa_human_old.gaf', sep='\t', header=None, names=names, skiprows=12)
    # filter protein annotations with evidence IEA
    goa_human = goa_human.loc[goa_human['evidence_code'] != 'IEA']
    # filter negative protein annotations
    goa_human = goa_human.loc[goa_human['qualifier'] != 'NOT']

    go_terms = get_go_terms()
    if include_parents:
        go_relations = get_go_terms_relationships()
        go_terms_with_parents = dict()
        print("(" + time.strftime("%c") + ")  Getting parents for every GO term...")
        for go_id in go_terms:
            go_terms_with_parents[go_id] = get_all_go_terms(go_id, go_relations)

    print("(" + time.strftime("%c") + ")  Filling protein annotations...")
    protein_annotations = dict()
    for index, row in goa_human.iterrows():

        if (row['db_object_id'] not in string_aliases) or (row['go_id'] not in go_terms):
            continue

        protein_id = string_aliases[row['db_object_id']]
        go_id = row['go_id']

        if include_parents:
            if protein_id not in protein_annotations:
                protein_annotations[protein_id] = go_terms_with_parents[go_id]
            else:
                protein_annotations[protein_id].update(go_terms_with_parents[go_id])
        else:
            if protein_id not in protein_annotations:
                protein_annotations[protein_id] = {go_id}
            else:
                protein_annotations[protein_id].update({go_id})

    return protein_annotations


def save_go_protein_annotations(file_interactions, protein_go_annotations, write_file):
    """ Save protein go annotations of proteins in the ppi file

    :param file_interactions: file that contains the protein interactions
    :type file_interactions: str
    :param protein_go_annotations:
    :type protein_go_annotations: dict
    :param write_file: file to write the protein go annotations
    :type write_file: str
    :return: None
    """
    human_ppi = pd.read_csv(file_interactions, header=0, sep='\t')
    proteins = set(human_ppi['protein1'].values.tolist() + human_ppi['protein2'].values.tolist())
    with open(write_file, 'w') as file:
        file.write('protein_id\tgo_id\n')
        for protein in proteins:
            if protein in protein_go_annotations:
                go_annotations = protein_go_annotations[protein]
                for go_id in go_annotations:
                    file.write(protein + '\t' + go_id + '\n')


def save_go_protein_annotations_with_parents():
    annotations = get_protein_go_annotations()
    print("(" + time.strftime("%c") + ")  Writing into HumanPPI700_GO...")
    save_go_protein_annotations('../data/human_ppi_700/HumanPPI.txt', annotations,
                                '../data/human_ppi_700/HumanPPI_GO.txt')
    print("(" + time.strftime("%c") + ")  Writing into HumanPPI900_GO...")
    save_go_protein_annotations('../data/human_ppi_900/HumanPPI.txt', annotations,
                                '../data/human_ppi_900/HumanPPI_GO.txt')


def save_go_protein_annotations_no_parents():
    annotations = get_protein_go_annotations(include_parents=False)
    print("(" + time.strftime("%c") + ")  Writing into HumanPPI700_GO...")
    save_go_protein_annotations('../data/human_ppi_700/HumanPPI.txt', annotations,
                                '../data/human_ppi_700/HumanPPI_GO_no_parents.txt')
    print("(" + time.strftime("%c") + ")  Writing into HumanPPI900_GO...")
    save_go_protein_annotations('../data/human_ppi_900/HumanPPI.txt', annotations,
                                '../data/human_ppi_900/HumanPPI_GO_no_parents.txt')


if __name__ == '__main__':
    save_go_protein_annotations_with_parents()
    # save_go_protein_annotations_no_parents()
