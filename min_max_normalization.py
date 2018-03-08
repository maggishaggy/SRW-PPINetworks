import pandas as pd
from sklearn import preprocessing
import numpy as np


def load_matrix(file_name):
    return pd.read_table(file_name, sep=',')


def write_matrix(x, file_name):
    x.to_csv(file_name, sep=',', index=False)


def normalize(file_name_from, file_name_to):
    matrix = load_matrix(file_name_from)
    for column in matrix.columns:
        if column not in ['protein1', 'protein2']:
            x = matrix[[column]].values.astype(float)
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)[:, 0]
            x_scaled = pd.Series(np.around(x_scaled, 5))
            matrix[column] = x_scaled
    write_matrix(matrix, file_name_to)


HumanPPI700_interactions_BP = 'Data/HumanPPI700_interactions_BP.txt'
HumanPPI700_interactions_BP_normalized = 'Data/HumanPPI700_interactions_BP_normalized.txt'
HumanPPI700_interactions_MF = 'Data/HumanPPI700_interactions_MF.txt'
HumanPPI700_interactions_MF_normalized = 'Data/HumanPPI700_interactions_MF_normalized.txt'
HumanPPI700_interactions_CC = 'Data/HumanPPI700_interactions_CC.txt'
HumanPPI700_interactions_CC_normalized = 'Data/HumanPPI700_interactions_CC_normalized.txt'
HumanPPI900_interactions_BP = 'Data/HumanPPI900_interactions_BP.txt'
HumanPPI900_interactions_BP_normalized = 'Data/HumanPPI900_interactions_BP_normalized.txt'
HumanPPI900_interactions_MF = 'Data/HumanPPI900_interactions_MF.txt'
HumanPPI900_interactions_MF_normalized = 'Data/HumanPPI900_interactions_MF_normalized.txt'
HumanPPI900_interactions_CC = 'Data/HumanPPI900_interactions_CC.txt'
HumanPPI900_interactions_CC_normalized = 'Data/HumanPPI900_interactions_CC_normalized.txt'

normalize(HumanPPI700_interactions_BP, HumanPPI700_interactions_BP_normalized)
normalize(HumanPPI700_interactions_MF, HumanPPI700_interactions_MF_normalized)
normalize(HumanPPI700_interactions_CC, HumanPPI700_interactions_CC_normalized)
normalize(HumanPPI900_interactions_BP, HumanPPI900_interactions_BP_normalized)
normalize(HumanPPI900_interactions_MF, HumanPPI900_interactions_MF_normalized)
normalize(HumanPPI900_interactions_CC, HumanPPI900_interactions_CC_normalized)
