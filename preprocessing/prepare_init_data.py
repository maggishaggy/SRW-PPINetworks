"""
Step 1
This file does the pre-processing of the 9606.protein.links file.
"""
import time
import pandas as pd


def read_protein_interactions_data(input_file, output_file1, output_file2, output_file3):
    """ Reads the txt file of String-db protein interactions and filters them
    with combined_score column with values 700 and 900

    :param input_file: file path of the 9606.protein.links file
    :param output_file1: file path to save the protein interactions
    :param output_file2: file path to save the filtered (700) protein interactions
    :param output_file3: file path to save the filtered (900) protein interactions
    :return: None
    """
    print("(" + time.strftime("%c") + ")  Reading Protein Links file...")
    data = pd.read_csv(input_file, header=0, sep=' ')
    ppi_700 = data.loc[data['combined_score'] >= 700]
    ppi_900 = data.loc[data['combined_score'] >= 900]
    print("(" + time.strftime("%c") + ")  Writing into HumanPPI, HumanPPI700 & HumanPPI900...")
    data.to_csv(output_file1, index=None, sep='\t')
    ppi_700.to_csv(output_file2, index=None, sep='\t')
    ppi_900.to_csv(output_file3, index=None, sep='\t')


if __name__ == '__main__':
    read_protein_interactions_data('../data/9606.protein.links.detailed.v10.5.txt',
                                   '../data/HumanPPI.txt',
                                   '../data/human_ppi_700/HumanPPI.txt',
                                   '../data/human_ppi_900/HumanPPI.txt')

