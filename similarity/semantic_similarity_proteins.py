import time
import numpy as np
import pandas as pd
from igraph import *


def get_proteins_from_largest_connected_component(interactions):
    """ Finds the proteins that are part of the largest connected components of the graph

    :param interactions: protein-protein interactions
    :type interactions: pandas.DataFrame
    :return: list of vertices from the largest connected components
    :rtype: list
    """
    graph = Graph(directed=False)
    vertices = np.unique(np.concatenate((interactions[['protein1']].values,
                                         interactions[['protein2']].values))).tolist()
    graph.add_vertices(vertices)
    graph.add_edges(interactions[['protein1', 'protein2']].values)

    components = graph.components()
    lcc = components.giant()

    return lcc.vs['name']


def combine_scores(sim_mat, method):
    """ Combines the scores using rcmax or BMA method

    :param sim_mat: similarity score matrix
    :type sim_mat: np.array
    :param method: method for combining the scores, on of 'rcmax' or 'BMA'
    :type method: str
    :return: combined score
    :rtype: float
    """
    if method not in ['rcmax', 'BMA']:
        Exception('Method must be rcmax or BMA')

    if method == 'rcmax':
        row_score = np.mean(np.nanmax(sim_mat, axis=1))
        col_score = np.mean(np.nanmax(sim_mat, axis=0))
        return max(row_score, col_score)
    else:
        row_max = np.nanmax(sim_mat, axis=0)
        col_max = np.nanmax(sim_mat, axis=1)
        return np.sum(np.concatenate((row_max, col_max))) / np.sum(sim_mat.shape)


def protein_similarities(annotations, interactions, go_similarities, column_name, method):
    """ Calculate protein-protein similarities for all interactions

    :param annotations: protein go term annotations dictionary
    :type annotations: dict
    :param interactions: data frame containing the protein-protein interactions
    :type interactions: pandas.DataFrame
    :param go_similarities: go term semantic similarities
    :type go_similarities: pandas.DataFrame
    :param column_name: name of the new added column
    :type column_name: str
    :param method: method for combining the scores, on of 'rcmax' or 'BMA'
    :type method: str
    :return: the interactions data frame with added column with similarities
    :rtype: pandas.DataFrame
    """
    protein1 = interactions.protein1.values.tolist()
    protein2 = interactions.protein2.values.tolist()

    interactions[column_name] = [combine_scores(go_similarities.ix[annotations[p1], annotations[p2]].values, method)
                                 if p1 in annotations and p2 in annotations else np.nan
                                 for p1, p2 in zip(protein1, protein2)]

    return interactions


def calc_protein_similarities(type_filtering, go_type):
    """ Calculates and saves the ppi with added several protein similarity attributes

    :param type_filtering: the type of filtering of the interactions network
    :type type_filtering: str
    :param go_type: type of the GO
    :type go_type: str
    :return: None
    """
    interactions_file = f'../data/human_ppi_{type_filtering}/HumanPPI.txt'
    interactions = pd.read_table(interactions_file, header=0)
    lcc_proteins = get_proteins_from_largest_connected_component(interactions)
    interactions = interactions.join(pd.DataFrame({'protein1': lcc_proteins}).set_index('protein1'),
                                     how='inner', on='protein1')
    annotations_file = f'../data/human_ppi_700/HumanPPI_GO_{go_type}_no_bias.txt'
    annotations = pd.read_table(annotations_file, header=0, names=['PID', 'GO'])
    annotations = annotations.groupby('PID')['GO'].apply(list).to_dict()
    go_similarities_files = [f'../data/sim/human_ppi_{type_filtering}/GO_{go_type}_no_bias_resnik.txt',
                             f'../data/sim/human_ppi_{type_filtering}/GO_{go_type}_no_bias_resnik.txt',
                             f'../data/sim/human_ppi_{type_filtering}/GO_{go_type}_no_bias_lin.txt',
                             f'../data/sim/human_ppi_{type_filtering}/GO_{go_type}_no_bias_lin.txt',
                             f'../data/sim/human_ppi_{type_filtering}/GO_{go_type}_no_bias_wang.txt',
                             f'../data/sim/human_ppi_{type_filtering}/GO_{go_type}_no_bias_wang.txt']
    column_names = ['resnik_rcmax', 'resnik_bma', 'lin_rcmax',
                    'lin_bma', 'wang_rcmax', 'wang_bma']
    methods = ['rcmax', 'BMA', 'rcmax',
               'BMA', 'rcmax', 'BMA']
    print('Computing protein similarities ...')
    start_time = time.time()
    for file_name, column_name, method in zip(go_similarities_files, column_names, methods):
        print(f'Computing {column_name} protein similarities ...')
        go_similarities = pd.read_csv(file_name, sep='\t', header=0, index_col=0)
        interactions = protein_similarities(annotations, interactions, go_similarities, column_name, method)
        interactions.to_csv(f'../data/final/human_ppi_{type_filtering}/HumanPPI_{go_type}_no_bias.txt',
                            sep='\t', index=False)
    end_time = time.time()
    print(end_time - start_time)
    print('Done.')
    

if __name__ == '__main__':
    calc_protein_similarities('700', 'BP')
    calc_protein_similarities('700', 'MF')
    calc_protein_similarities('700', 'CC')
    # calc_protein_similarities('900', 'BP')
    # calc_protein_similarities('900', 'MF')
    # calc_protein_similarities('900', 'CC')
