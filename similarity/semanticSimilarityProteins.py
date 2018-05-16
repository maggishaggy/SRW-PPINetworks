import pandas as pd
import numpy as np
import time
import math


def combine_scores(sim_mat, method):
    """ Combines the scores using rcmax or BMA method

    :param sim_mat: similarity score matrix
    :type sim_mat: np.ndarray
    :param method: method for combining the scores, on of 'rcmax' or 'BMA'
    :type method: str
    :return: combined score
    :rtype: float
    """
    if method not in ['rcmax', 'BMA']:
        Exception('Method must be rcmax or BMA')

    if method == 'rcmax':
        row_score = np.mean(np.max(sim_mat, axis=1))
        col_score = np.mean(np.max(sim_mat, axis=0))
        return max(row_score, col_score)
    else:
        row_max = np.max(sim_mat, axis=0)
        col_max = np.max(sim_mat, axis=1)
        return np.sum(np.concatenate((row_max, col_max))) / np.sum(sim_mat.shape)


def half_vec_to_symmetric_matrix(half_vec):
    """Transforms a half-vectorization vector of the lower triangular part of matrix without the diagonal
    elements into the whole symmetric matrix

    :param half_vec: a numpy ndarray with shape (n,) representing the half-vectorization vector
    :type half_vec: np.ndarray
    :return: the symmetric matrix
    :rtype: numpy.ndarray
    """
    half_vec = half_vec.astype('float')
    k = -2 * half_vec.shape[0]
    d = max(int(1 + math.sqrt(1-4*k)) // 2, int(1 - math.sqrt(1-4*k)) // 2)
    matrix = np.zeros((d, d))
    index = 0
    for i in range(d):
        matrix[i][i] = 1
        for j in range(i+1, d):
            matrix[i][j] = half_vec[index]
            matrix[j][i] = half_vec[index]
            index += 1
    return matrix


def calc_protein_similarity_matrix(input_file1, input_file2, method, output_file):
    """ Saves a semantic similarity matrix of the GO terms in the given dataset of GO protein
    annotations and GO terms semantic similarity

    :param input_file1: path to the tab separated file containing GO protein annotations
    :type input_file1: str
    :param input_file2: path to the tab separated file containing the similarity matrix of GO terms
    :type: input_file2: str
    :param method: combine method, one of rcmax or BMA
    :type method: str
    :param output_file: path to save the similarity matrix
    :type output_file: str
    :return: None
    """
    goa = pd.read_table(input_file1, header=None, names=['PID', 'PID2', 'GO', ''])
    proteins = np.unique(goa.PID.values).tolist()
    sim_matrix = pd.read_table(input_file2, header=0, index_col=0)
    
    num_proteins = len(proteins)
    
    print('Computing protein similarities ...')
    start_time = time.time()
    sim_vec = [combine_scores(sim_matrix.filter(items=goa[goa.PID == proteins[i]].GO.values.tolist(),
                                                axis=0).filter(items=goa[goa.PID == proteins[j]].GO.values.tolist(),
                                                               axis=1).as_matrix(), method)
               for i in range(num_proteins) for j in range(i+1, num_proteins)]
    scores = pd.DataFrame(half_vec_to_symmetric_matrix(np.array(sim_vec)), index=proteins, columns=proteins)
    end_time = time.time()
    print(end_time - start_time)
    scores.to_csv(output_file, sep='\t', header=True, index=True)
    print('Done.')
    

if __name__ == '__main__':
    calc_protein_similarity_matrix('../../Data/GO annotations/HumanPPI700_GO_BP_modified.txt',
                                   '../../Data/Semantic similarity/700modified_BP_resnik.txt',
                                   'rcmax',
                                   '../../Data/Protein semantic similarity/HumanPPI700_GO_BP_modified_resnik_rcmax.txt')
    calc_protein_similarity_matrix('../../Data/GO annotations/HumanPPI700_GO_BP_modified.txt',
                                   '../../Data/Semantic similarity/700modified_BP_resnik.txt',
                                   'BMA',
                                   '../../Data/Protein semantic similarity/HumanPPI700_GO_BP_modified_resnik_BMA.txt')
