"""
This script gives statistics for every BP, MF and CC go term number of annotations
"""
import pandas as pd
import matplotlib.pyplot as plt


def get_number_of_annotations(file_name, output_file, histogram_file=None):
    """ Calculates the number of annotations for every go term in the annotations file

    :param file_name: file path to the file with protein-go annotations
    :type file_name: str
    :param output_file: file path to save the result
    :type output_file: str
    :param histogram_file: file path to save the histogram
    :type histogram_file: str
    :return: None
    """
    counts = pd.read_csv(file_name, sep='\t', header=0).groupby(['go_id']).count()
    counts.columns = ['protein_count']
    counts.to_csv(output_file, sep='\t')
    if histogram_file is not None:
        counts.plot(kind='bar', legend=False, rot=90, fontsize=3, stacked=True, width=1)
        plt.savefig(histogram_file)


if __name__ == '__main__':
    get_number_of_annotations('../data/human_ppi_700/HumanPPI_GO_BP_no_bias.txt',
                              '../data/human_ppi_700/HumanPPI_GO_BP_no_bias_count.txt')
    get_number_of_annotations('../data/human_ppi_700/HumanPPI_GO_MF_no_bias.txt',
                              '../data/human_ppi_700/HumanPPI_GO_MF_no_bias_count.txt')
    get_number_of_annotations('../data/human_ppi_700/HumanPPI_GO_CC_no_bias.txt',
                              '../data/human_ppi_700/HumanPPI_GO_CC_no_bias_count.txt')
