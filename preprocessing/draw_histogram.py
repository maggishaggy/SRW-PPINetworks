import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def draw_hist(hist_file, result_file):
    data = pd.read_csv(hist_file, sep='\t', header=0, index_col=0, dtype={'minimum number of proteins': np.int32})
    data = data[data['minimum number of proteins'] > -1]
    ax = sns.countplot(x='minimum number of proteins', data=data)
    plt.savefig(result_file)
    plt.show()


if __name__ == '__main__':
    filtering_type = '700'
    onto = 'CC'
    file_histogram = f'../data/human_ppi_{filtering_type}/HumanPPI_GO_{onto}_histogram.txt'
    draw_hist(file_histogram, f'min_protein_num_{filtering_type}_{onto}.jpg')
