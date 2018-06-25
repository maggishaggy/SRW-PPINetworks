import pandas as pd
from pulp import LpVariable, LpProblem, LpMinimize, LpAffineExpression, value
import numpy as np


def load_annotations(file_name_t1, file_name_t2):
    """ Load protein - term annotation data

    :param file_name_t1: name of the file containing t1 protein - term annotations
    :type file_name_t1: str
    :param file_name_t2: name of the file containing t2 protein - term annotations
    :type file_name_t2: str
    :return: protein - term annotation data
    :rtype: dict
    """
    df = pd.read_table(file_name_t1, sep='\t')
    annotations_t1 = dict()
    annotations_t2 = dict()
    for index, row in df.iterrows():
        if row['protein_id'] not in annotations_t1.keys():
            annotations_t1[row['protein_id']] = [row['go_id']]
        else:
            annotations_t1[row['protein_id']].append(row['go_id'])
    df = pd.read_table(file_name_t2, sep='\t')
    for index, row in df.iterrows():
        if row['protein_id'] not in annotations_t2.keys():
            annotations_t2[row['protein_id']] = [row['go_id']]
        else:
            annotations_t2[row['protein_id']].append(row['go_id'])
    return annotations_t1, annotations_t2


def solve_set_covering(num_terms, protein):
    """ Find minimum number of proteins needed to reconstruct annotation set for a given protein using integer
        programming - set covering problem

    :param num_terms: number of terms
    :type num_terms: int
    :param protein: protein annotations data
    :type protein: list
    :return: minimum number of proteins needed to reconstruct annotation set
    :rtype: np.float
    """
    c = [1 for i in range(num_terms)]
    x_name = ['x_' + str(i + 1) for i in range(num_terms)]
    x = [LpVariable(name=x_name[i], lowBound=0, upBound=1, cat='Binary') for i in range(num_terms)]
    problem = LpProblem('set_covering', LpMinimize)
    z = LpAffineExpression([(x[i], c[i]) for i in range(num_terms)])
    for i in protein:
        problem += sum([x[j] for j in i]) >= 1
    problem += z
    problem.solve()
    return np.sum([value(i) for i in x])


def create_data(annotations_t1, annotations_t2, p):
    """ Calculate number of terms for protein p and create dictionary with protein - term annotations containing
        only terms of p

    :param annotations_t1: protein - term annotation data for t1
    :type annotations_t1: dict
    :param annotations_t2: protein - term annotation data for t2
    :type annotations_t2: dict
    :param p: protein id
    :type p: str
    :return: number of terms for protein p and dictionary with protein - term annotations containing only terms of p
    """
    terms = annotations_t2[p]
    t_to_id = dict()
    for (t, i) in zip(terms, range(len(terms))):
        t_to_id[t] = i
    protein_data = []
    for p1 in annotations_t1.keys():
        if not p1 == p:
            terms1 = annotations_t1[p1]
            l = list()
            for term in terms1:
                if term in terms:
                    l.append(t_to_id[term])
            if len(l) > 0:
                protein_data.append(l)
    return len(terms), protein_data


def find_minimal_union(t1, t2, hist):
    """ For each protein, find minimum number of proteins needed to reconstruct its annotation set and write into file

    :param t1: name of the file containing t1 protein - term annotations
    :type t1: str
    :param t2: name of the file containing t2 protein - term annotations
    :type t2: str
    :param hist: name of the file containing the histogram
    :type hist: str
    """
    annotations_t1, annotations_t2 = load_annotations(t1, t2)
    histogram = dict()
    for (p, i) in zip(annotations_t1.keys(), range(len(annotations_t1.keys()))):
        if p not in annotations_t2.keys():
            histogram[p] = -1
            continue
        num_terms, protein_data = create_data(annotations_t1, annotations_t2, p)
        m = solve_set_covering(num_terms, protein_data)
        histogram[p] = m
        if i % 100 == 0:
            print('Processed: ' + str(i) + '/' + str(len(annotations_t1.keys())))
    with open(hist, 'w+') as doc:
        doc.write('protein_id\tminimum number of proteins\n')
        for key in histogram.keys():
            doc.write(key + '\t' + str(histogram[key]) + '\n')


if __name__ == '__main__':
    filtering_type = '900'
    onto = 'BP'
    t1_annotations = f'data/human_ppi_{filtering_type}/HumanPPI_GO_{onto}_no_bias.txt'
    t2_annotations = f'data/human_ppi_{filtering_type}/t2/HumanPPI_GO_{onto}_no_bias.txt'
    minimal_union_histogram = f'{directory}data/human_ppi_{filtering_type}/HumanPPI_GO_{onto}_histogram.txt'
    find_minimal_union(t1_annotations, t2_annotations, minimal_union_histogram)
