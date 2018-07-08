import numpy as np
import pandas as pd
from igraph import *
import seaborn as sns
import matplotlib.pyplot as plt


def plot_ppi_network_stats(file_interactions, filter_type):
    """ Plots and saves PPI network visualizations

    :param file_interactions: name of the file with protein-protein interactions
    :type file_interactions: str
    :param filter_type: current filter type of the protein interactions (700 or 900)
    :type filter_type: str
    :return: None
    """
    sns.set(style='whitegrid', context='poster', font='DejaVu Sans', font_scale=1.5)

    if os.path.exists(file_interactions):
        ppi = pd.read_csv(file_interactions, header=0, sep='\t').fillna(0)
        scores = ppi.combined_score.values
        ppi.combined_score = (scores - int(filter_type)) / (1000 - int(filter_type))
    else:
        raise Exception('{} does not exist'.format(file_interactions))

    graph = Graph(directed=False)
    vertices = np.unique(np.concatenate((ppi[['protein1']].values, ppi[['protein2']].values))).tolist()
    graph.add_vertices(vertices)
    graph.add_edges(ppi[['protein1', 'protein2']].values)
    for feature in ppi.columns[9:]:
        graph.es[feature] = ppi[feature].values.tolist()

    del ppi
    bins = graph.degree_distribution()._bins
    data = pd.DataFrame()
    data['degree'] = np.array([y for y in bins])
    data['number of proteins'] = np.array([x for x in range(1, len(bins)+1)])
    f, ax = plt.subplots()
    ax.set(xscale="log", yscale="log")
    ax = sns.regplot('degree', 'number of proteins', data, ax=ax,
                     scatter_kws={"s": 200}, fit_reg=False, color='xkcd:lavender')
    ax.grid(False)
    plt.title(f'Degree distribution - Human PPI {filter_type} network')
    plt.savefig(f'degree_distribution_{filter_type}.png')
    plt.show()
    data = pd.DataFrame()
    path_lenghts = np.array(graph.shortest_paths())
    max_length = np.max(path_lenghts)
    percent = np.zeros((max_length + 1))
    for i in range(max_length + 1):
        percent[i] = np.sum(path_lenghts.flatten() == i) / np.sum(path_lenghts.flatten() >= 0)
    data['shortest path'] = np.array([x for x in range(max_length + 1)])
    data['percent'] = percent
    ax = sns.regplot('shortest path', 'percent', data,
                     scatter_kws={"s": 200}, fit_reg=False, color='xkcd:lavender')
    ax.grid(False)
    plt.title(f'Shortest path Human PPI {filter_type} network')
    plt.savefig(f'shortest_path_{filter_type}.png')
    plt.show()


def plot_annotations_num_levels(ont, filter_type):
    """ Plots graphs for number of annotations with respect to the level of the annotation GO term

    :param ont: name of the current ontology (BP, CC or MF)
    :type ont: str
    :param filter_type: current filter type of the protein interactions (700 or 900)
    :type filter_type: str
    :return: None
    """
    sns.set(style='white', context='poster', font='DejaVu Sans')
    data1 = pd.read_csv(f'data/human_ppi_{filter_type}/HumanPPI_GO_{ont}_no_bias_level_count.txt',
                        sep='\t', header=0)
    data1['time'] = 't1'
    data2 = pd.read_csv(f'data/human_ppi_{filter_type}/t2/HumanPPI_GO_{ont}_no_bias_level_count.txt',
                        sep='\t', header=0)
    data2['time'] = 't2'
    data = pd.concat([data1, data2])
    g = sns.factorplot(x="level", y="number of annotations", hue="time", data=data,
                       saturation=.5, kind="bar", ci=None, aspect=2, color='#e31a1c')
    plt.title(f'{ont} ontology')
    plt.savefig(f'annotations_levels_count_{filter_type}_{ont}.png')
    plt.show()


if __name__ == '__main__':
    filtering_type = '700'
    onto = 'BP'
    file = f'data/final/human_ppi_{filtering_type}/HumanPPI_{onto}_no_bias.txt'
    plot_ppi_network_stats(file, filtering_type)
    plot_annotations_num_levels(onto, filtering_type)
