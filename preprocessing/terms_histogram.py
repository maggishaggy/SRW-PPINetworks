"""
This script gives statistics for every BP, MF and CC go term number of annotations
"""
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt


def terms_levels_number_of_annotations(ont, filter_type, time_step):
    if os.path.exists(f'../data/go/{ont}GOtermLevels.pkl'):
        with open(f'../data/go/{ont}GOtermLevels.pkl', 'rb') as f:
            go_terms_levels = pickle.load(f)
    else:
        go_relations = pd.read_csv(f'../data/go/{ont}GOfull.txt', header=None, sep=' ',
                                   names=['child', 'relation', 'parent'])
        # filter for child-parent relations
        go_relations = go_relations.loc[((go_relations['relation'] == 'is_a') |
                                         (go_relations['relation'] == 'part_of'))]
        level = 0
        go_terms_levels = dict()
        current_level = set(go_relations['parent'].values.tolist()).difference(go_relations['child'].values.tolist())
        while True:
            for term in current_level:
                go_terms_levels[term] = level
            level += 1
            current_level = set(go_relations[go_relations['parent'].isin(current_level).values]
                                ['child'].values.tolist())
            if len(current_level) == 0:
                break
        with open(f'../data/go/{ont}GOtermLevels.pkl', 'wb') as f:
            pickle.dump(go_terms_levels, f, pickle.HIGHEST_PROTOCOL)
    annotations = pd.read_csv(f'../data/human_ppi_{filter_type}/{time_step}HumanPPI_GO_{ont}_no_bias.txt',
                              header=0, sep='\t')['go_id'].values.tolist()
    annotations_levels = dict()
    for term in annotations:
        if go_terms_levels[term] in annotations_levels:
            annotations_levels[go_terms_levels[term]] += 1
        else:
            annotations_levels[go_terms_levels[term]] = 1
    annotations_levels = pd.DataFrame({'level': list(annotations_levels.keys()),
                                       'number of annotations': list(annotations_levels.values())})
    annotations_levels.to_csv(f'../data/human_ppi_{filter_type}/{time_step}HumanPPI_GO_{ont}_no_bias_level_count.txt',
                              sep='\t', index=False)


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
    get_number_of_annotations('../data/human_ppi_700/HumanPPI_GO_BP.txt',
                              '../data/human_ppi_700/HumanPPI_GO_BP_count.txt')
    get_number_of_annotations('../data/human_ppi_700/HumanPPI_GO_MF.txt',
                              '../data/human_ppi_700/HumanPPI_GO_MF_count.txt')
    get_number_of_annotations('../data/human_ppi_700/HumanPPI_GO_CC.txt',
                              '../data/human_ppi_700/HumanPPI_GO_CC_count.txt')
    terms_levels_number_of_annotations('BP', '700', '')
    terms_levels_number_of_annotations('BP', '700', 't2/')
    terms_levels_number_of_annotations('MF', '700', '')
    terms_levels_number_of_annotations('MF', '700', 't2/')
    terms_levels_number_of_annotations('CC', '700', '')
    terms_levels_number_of_annotations('CC', '700', 't2/')
