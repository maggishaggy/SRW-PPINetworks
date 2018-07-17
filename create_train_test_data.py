import numpy as np
import pandas as pd


def create_train_test_data(proteins_k, proteins_lk, proteins_nk, train_file, test_file):
    """ Divide proteins into train and test sets.

    :param proteins_k: proteins that are part of the knowledge data
    :type proteins_k: numpy.array
    :param proteins_lk: proteins that are part of the limited-knowledge data
    :type proteins_lk: numpy.array
    :param proteins_nk: proteins that are part of the no-knowledge data
    :type proteins_nk: numpy.array
    :param train_file: file path to save the train set
    :type train_file: str
    :param test_file: file path to test the train set
    :type test_file: str
    :return: None
    """
    np.random.shuffle(proteins_k)
    np.random.shuffle(proteins_lk)
    np.random.shuffle(proteins_nk)

    with open(train_file, 'w') as train_f, open(test_file, 'w') as test_f:
        train_f.write('protein_id\n')
        test_f.write('protein_id\n')

    with open(train_file, 'ab') as train_f, open(test_file, 'ab') as test_f:
        for proteins in [proteins_k, proteins_lk, proteins_nk]:
            length = proteins.shape[0]
            train = proteins[:round(length*0.7)]
            np.savetxt(train_f, train, fmt='%s')
            test = proteins[round(length*0.7):]
            np.savetxt(test_f, test, fmt='%s')


if __name__ == '__main__':
    for ontology_type in ['BP', 'MF', 'CC']:
        for filtering_type in ['700', '900']: 
            prot_k = pd.read_csv(f'data/final/human_ppi_{filtering_type}/{ontology_type}_no_bias_K.txt',
                                 header=0, sep='\t').protein_id.values
            prot_nk = pd.read_csv(f'data/final/human_ppi_{filtering_type}/{ontology_type}_no_bias_NK.txt',
                                  header=0, sep='\t').protein_id.values
            prot_lk = pd.read_csv(f'data/final/human_ppi_{filtering_type}/{ontology_type}_no_bias_LK.txt',
                                  header=0, sep='\t').protein_id.values
            create_train_test_data(prot_k, prot_lk, prot_nk,
                                   f'data/final/human_ppi_{filtering_type}/train_{ontology_type}_no_bias.txt',
                                   f'data/final/human_ppi_{filtering_type}/test_{ontology_type}_no_bias.txt')
