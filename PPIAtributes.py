import pandas as pd
import numpy as np
import time


def read_matrix(file_name):
    matrix = pd.read_table(file_name)
    return matrix


def add_attribute(file_name, matrix, attr_name):
    x = pd.read_table(file_name, sep=',')
    attr = pd.Series(np.zeros(x.shape[0]))
    x = x.assign(attr_name=attr.values)
    col_names = x.columns.values
    col_names[len(col_names) - 1] = attr_name
    x.columns = col_names
    missing_interactions = 0
    start = time.clock()
    m = np.matrix(matrix.as_matrix()).mean()
    for i in range(x.shape[0]):
        if x.at[i, 'protein1'] in matrix and x.at[i, 'protein2'] in matrix:
            x.at[i, attr_name] = round(matrix.at[x.at[i, 'protein1'], x.at[i, 'protein2']] + m, 5)
        else:
            missing_interactions += 1
        # print str(i) + '/' + str(x.shape[0])
    elapsed = time.clock() - start
    print 'Graph: ' + file_name
    print 'Measure: ' + attr_name
    print 'Elapsed time: ' + str(round(elapsed, 5))
    print 'Number of interactions for proteins which are not in the similarity matrix: ' + str(missing_interactions)
    x.to_csv(file_name, sep=',', index=False)


def add_attributes_PPI700_BP():
    HumanPPI700_interactions_BP = 'Data/Protein-Protein interactions/HumanPPI700_interactions_BP.txt'
    HumanPPI700_GO_BP_modified_lin_BMA = 'Data/HumanPPI700_GO_BP_modified_lin_BMA.txt'
    HumanPPI700_GO_BP_modified_lin_rcmax = 'Data/HumanPPI700_GO_BP_modified_lin_rcmax.txt'
    HumanPPI700_GO_BP_modified_resnik_BMA = 'Data/HumanPPI700_GO_BP_modified_resnik_BMA.txt'
    HumanPPI700_GO_BP_modified_resnik_rcmax = 'Data/HumanPPI700_GO_BP_modified_resnik_rcmax.txt'
    HumanPPI700_GO_BP_modified_wang_BMA = 'Data/HumanPPI700_GO_BP_modified_wang_BMA.txt'
    HumanPPI700_GO_BP_modified_wang_rcmax = 'Data/HumanPPI700_GO_BP_modified_wang_rcmax.txt'
    similarity_matrix = read_matrix(HumanPPI700_GO_BP_modified_lin_BMA)
    add_attribute(HumanPPI700_interactions_BP, similarity_matrix, 'lin_BMA')
    similarity_matrix = read_matrix(HumanPPI700_GO_BP_modified_lin_rcmax)
    add_attribute(HumanPPI700_interactions_BP, similarity_matrix, 'lin_rcmax')
    similarity_matrix = read_matrix(HumanPPI700_GO_BP_modified_resnik_BMA)
    add_attribute(HumanPPI700_interactions_BP, similarity_matrix, 'resnik_BMA')
    similarity_matrix = read_matrix(HumanPPI700_GO_BP_modified_resnik_rcmax)
    add_attribute(HumanPPI700_interactions_BP, similarity_matrix, 'resnik_rcmax')
    similarity_matrix = read_matrix(HumanPPI700_GO_BP_modified_wang_BMA)
    add_attribute(HumanPPI700_interactions_BP, similarity_matrix, 'wang_BMA')
    similarity_matrix = read_matrix(HumanPPI700_GO_BP_modified_wang_rcmax)
    add_attribute(HumanPPI700_interactions_BP, similarity_matrix, 'wang_rcmax')


def add_attributes_PPI700_MF():
    HumanPPI700_interactions_MF = 'Data/HumanPPI700_interactions_MF.txt'
    HumanPPI700_GO_MF_modified_lin_BMA = 'Data/HumanPPI700_GO_MF_modified_lin_BMA.txt'
    HumanPPI700_GO_MF_modified_lin_rcmax = 'Data/HumanPPI700_GO_MF_modified_lin_rcmax.txt'
    HumanPPI700_GO_MF_modified_resnik_BMA = 'Data/HumanPPI700_GO_MF_modified_resnik_BMA.txt'
    HumanPPI700_GO_MF_modified_resnik_rcmax = 'Data/HumanPPI700_GO_MF_modified_resnik_rcmax.txt'
    HumanPPI700_GO_MF_modified_wang_BMA = 'Data/HumanPPI700_GO_MF_modified_wang_BMA.txt'
    HumanPPI700_GO_MF_modified_wang_rcmax = 'Data/HumanPPI700_GO_MF_modified_wang_rcmax.txt'
    similarity_matrix = read_matrix(HumanPPI700_GO_MF_modified_lin_BMA)
    add_attribute(HumanPPI700_interactions_MF, similarity_matrix, 'lin_BMA')
    similarity_matrix = read_matrix(HumanPPI700_GO_MF_modified_lin_rcmax)
    add_attribute(HumanPPI700_interactions_MF, similarity_matrix, 'lin_rcmax')
    similarity_matrix = read_matrix(HumanPPI700_GO_MF_modified_resnik_BMA)
    add_attribute(HumanPPI700_interactions_MF, similarity_matrix, 'resnik_BMA')
    similarity_matrix = read_matrix(HumanPPI700_GO_MF_modified_resnik_rcmax)
    add_attribute(HumanPPI700_interactions_MF, similarity_matrix, 'resnik_rcmax')
    similarity_matrix = read_matrix(HumanPPI700_GO_MF_modified_wang_BMA)
    add_attribute(HumanPPI700_interactions_MF, similarity_matrix, 'wang_BMA')
    similarity_matrix = read_matrix(HumanPPI700_GO_MF_modified_wang_rcmax)
    add_attribute(HumanPPI700_interactions_MF, similarity_matrix, 'wang_rcmax')


def add_attributes_PPI700_CC():
    HumanPPI700_interactions_CC = 'Data/HumanPPI700_interactions_CC.txt'
    HumanPPI700_GO_CC_modified_lin_BMA = 'Data/HumanPPI700_GO_CC_modified_lin_BMA.txt'
    HumanPPI700_GO_CC_modified_lin_rcmax = 'Data/HumanPPI700_GO_CC_modified_lin_rcmax.txt'
    HumanPPI700_GO_CC_modified_resnik_BMA = 'Data/HumanPPI700_GO_CC_modified_resnik_BMA.txt'
    HumanPPI700_GO_CC_modified_resnik_rcmax = 'Data/HumanPPI700_GO_CC_modified_resnik_rcmax.txt'
    HumanPPI700_GO_CC_modified_wang_BMA = 'Data/HumanPPI700_GO_CC_modified_wang_BMA.txt'
    HumanPPI700_GO_CC_modified_wang_rcmax = 'Data/HumanPPI700_GO_CC_modified_wang_rcmax.txt'
    similarity_matrix = read_matrix(HumanPPI700_GO_CC_modified_lin_BMA)
    add_attribute(HumanPPI700_interactions_CC, similarity_matrix, 'lin_BMA')
    similarity_matrix = read_matrix(HumanPPI700_GO_CC_modified_lin_rcmax)
    add_attribute(HumanPPI700_interactions_CC, similarity_matrix, 'lin_rcmax')
    similarity_matrix = read_matrix(HumanPPI700_GO_CC_modified_resnik_BMA)
    add_attribute(HumanPPI700_interactions_CC, similarity_matrix, 'resnik_BMA')
    similarity_matrix = read_matrix(HumanPPI700_GO_CC_modified_resnik_rcmax)
    add_attribute(HumanPPI700_interactions_CC, similarity_matrix, 'resnik_rcmax')
    similarity_matrix = read_matrix(HumanPPI700_GO_CC_modified_wang_BMA)
    add_attribute(HumanPPI700_interactions_CC, similarity_matrix, 'wang_BMA')
    similarity_matrix = read_matrix(HumanPPI700_GO_CC_modified_wang_rcmax)
    add_attribute(HumanPPI700_interactions_CC, similarity_matrix, 'wang_rcmax')


def add_attributes_PPI900_BP():
    HumanPPI900_interactions_BP = 'Data/HumanPPI900_interactions_BP.txt'
    HumanPPI900_GO_BP_modified_lin_BMA = 'Data/HumanPPI900_GO_BP_modified_lin_BMA.txt'
    HumanPPI900_GO_BP_modified_lin_rcmax = 'Data/HumanPPI900_GO_BP_modified_lin_rcmax.txt'
    HumanPPI900_GO_BP_modified_resnik_BMA = 'Data/HumanPPI900_GO_BP_modified_resnik_BMA.txt'
    HumanPPI900_GO_BP_modified_resnik_rcmax = 'Data/HumanPPI900_GO_BP_modified_resnik_rcmax.txt'
    HumanPPI900_GO_BP_modified_wang_BMA = 'Data/HumanPPI900_GO_BP_modified_wang_BMA.txt'
    HumanPPI900_GO_BP_modified_wang_rcmax = 'Data/HumanPPI900_GO_BP_modified_wang_rcmax.txt'
    similarity_matrix = read_matrix(HumanPPI900_GO_BP_modified_lin_BMA)
    add_attribute(HumanPPI900_interactions_BP, similarity_matrix, 'lin_BMA')
    similarity_matrix = read_matrix(HumanPPI900_GO_BP_modified_lin_rcmax)
    add_attribute(HumanPPI900_interactions_BP, similarity_matrix, 'lin_rcmax')
    similarity_matrix = read_matrix(HumanPPI900_GO_BP_modified_resnik_BMA)
    add_attribute(HumanPPI900_interactions_BP, similarity_matrix, 'resnik_BMA')
    similarity_matrix = read_matrix(HumanPPI900_GO_BP_modified_resnik_rcmax)
    add_attribute(HumanPPI900_interactions_BP, similarity_matrix, 'resnik_rcmax')
    similarity_matrix = read_matrix(HumanPPI900_GO_BP_modified_wang_BMA)
    add_attribute(HumanPPI900_interactions_BP, similarity_matrix, 'wang_BMA')
    similarity_matrix = read_matrix(HumanPPI900_GO_BP_modified_wang_rcmax)
    add_attribute(HumanPPI900_interactions_BP, similarity_matrix, 'wang_rcmax')


def add_attributes_PPI900_MF():
    HumanPPI900_interactions_MF = 'Data/HumanPPI900_interactions_MF.txt'
    HumanPPI900_GO_MF_modified_lin_BMA = 'Data/HumanPPI900_GO_MF_modified_lin_BMA.txt'
    HumanPPI900_GO_MF_modified_lin_rcmax = 'Data/HumanPPI900_GO_MF_modified_lin_rcmax.txt'
    HumanPPI900_GO_MF_modified_resnik_BMA = 'Data/HumanPPI900_GO_MF_modified_resnik_BMA.txt'
    HumanPPI900_GO_MF_modified_resnik_rcmax = 'Data/HumanPPI900_GO_MF_modified_resnik_rcmax.txt'
    HumanPPI900_GO_MF_modified_wang_BMA = 'Data/HumanPPI900_GO_MF_modified_wang_BMA.txt'
    HumanPPI900_GO_MF_modified_wang_rcmax = 'Data/HumanPPI900_GO_MF_modified_wang_rcmax.txt'
    similarity_matrix = read_matrix(HumanPPI900_GO_MF_modified_lin_BMA)
    add_attribute(HumanPPI900_interactions_MF, similarity_matrix, 'lin_BMA')
    similarity_matrix = read_matrix(HumanPPI900_GO_MF_modified_lin_rcmax)
    add_attribute(HumanPPI900_interactions_MF, similarity_matrix, 'lin_rcmax')
    similarity_matrix = read_matrix(HumanPPI900_GO_MF_modified_resnik_BMA)
    add_attribute(HumanPPI900_interactions_MF, similarity_matrix, 'resnik_BMA')
    similarity_matrix = read_matrix(HumanPPI900_GO_MF_modified_resnik_rcmax)
    add_attribute(HumanPPI900_interactions_MF, similarity_matrix, 'resnik_rcmax')
    similarity_matrix = read_matrix(HumanPPI900_GO_MF_modified_wang_BMA)
    add_attribute(HumanPPI900_interactions_MF, similarity_matrix, 'wang_BMA')
    similarity_matrix = read_matrix(HumanPPI900_GO_MF_modified_wang_rcmax)
    add_attribute(HumanPPI900_interactions_MF, similarity_matrix, 'wang_rcmax')


def add_attributes_PPI900_CC():
    HumanPPI900_interactions_CC = 'Data/HumanPPI900_interactions_CC.txt'
    HumanPPI900_GO_CC_modified_lin_BMA = 'Data/HumanPPI900_GO_CC_modified_lin_BMA.txt'
    HumanPPI900_GO_CC_modified_lin_rcmax = 'Data/HumanPPI900_GO_CC_modified_lin_rcmax.txt'
    HumanPPI900_GO_CC_modified_resnik_BMA = 'Data/HumanPPI900_GO_CC_modified_resnik_BMA.txt'
    HumanPPI900_GO_CC_modified_resnik_rcmax = 'Data/HumanPPI900_GO_CC_modified_resnik_rcmax.txt'
    HumanPPI900_GO_CC_modified_wang_BMA = 'Data/HumanPPI900_GO_CC_modified_wang_BMA.txt'
    HumanPPI900_GO_CC_modified_wang_rcmax = 'Data/HumanPPI900_GO_CC_modified_wang_rcmax.txt'
    similarity_matrix = read_matrix(HumanPPI900_GO_CC_modified_lin_BMA)
    add_attribute(HumanPPI900_interactions_CC, similarity_matrix, 'lin_BMA')
    similarity_matrix = read_matrix(HumanPPI900_GO_CC_modified_lin_rcmax)
    add_attribute(HumanPPI900_interactions_CC, similarity_matrix, 'lin_rcmax')
    similarity_matrix = read_matrix(HumanPPI900_GO_CC_modified_resnik_BMA)
    add_attribute(HumanPPI900_interactions_CC, similarity_matrix, 'resnik_BMA')
    similarity_matrix = read_matrix(HumanPPI900_GO_CC_modified_resnik_rcmax)
    add_attribute(HumanPPI900_interactions_CC, similarity_matrix, 'resnik_rcmax')
    similarity_matrix = read_matrix(HumanPPI900_GO_CC_modified_wang_BMA)
    add_attribute(HumanPPI900_interactions_CC, similarity_matrix, 'wang_BMA')
    similarity_matrix = read_matrix(HumanPPI900_GO_CC_modified_wang_rcmax)
    add_attribute(HumanPPI900_interactions_CC, similarity_matrix, 'wang_rcmax')


add_attributes_PPI700_BP()
add_attributes_PPI700_MF()
add_attributes_PPI700_CC()
add_attributes_PPI900_BP()
add_attributes_PPI900_MF()
add_attributes_PPI700_CC()
