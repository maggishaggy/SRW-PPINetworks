import numpy as np
import pandas as pd
from igraph import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

from test import prepare_test_data


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
    sns.factorplot(x="level", y="number of annotations", hue="time", data=data,
                   saturation=.5, kind="bar", ci=None, aspect=2, color='#e31a1c')
    plt.title(f'{ont} ontology')
    plt.savefig(f'annotations_levels_count_{filter_type}_{ont}.png')
    plt.show()


def plot_aucs(aucs, filter_type, ont):
    x = sorted(aucs, reverse=True)
    y = list(range(len(x)))
    plt.figure()
    plt.style.use('seaborn-darkgrid')
    plt.fill_between(y, x, alpha=0.8, color='xkcd:lavender')
    plt.plot(y, x, 'o', ms=5, markerfacecolor="None",
             markeredgecolor='lightgray', markeredgewidth=1)
    plt.title(f'AUC - {ont} Ontology')
    plt.xlabel('Terms sorted by AUC')
    plt.ylabel('AUC')
    plt.ylim([0.0, 1.05])
    plt.savefig(f'data/trained/human_ppi_{filter_type}/auc_{ont}.jpg')


def plot_term_centric_performance_graph(x, y, filter_type, ont):
    plt.figure()
    sns.regplot(x=np.array(x), y=np.array(y), fit_reg=False, color='xkcd:lavender')
    plt.title(f'Term-centric performance - {ont}')
    plt.xlabel('Number of training samples with the term')
    plt.ylabel('Fmax measure')
    plt.savefig(f'data/trained/human_ppi_{filter_type}/function_centric_performance_{ont}.jpg')


def plot_term_centric_f_measure(term_results_file, skip_rows, file_test,
                                t1_file, file_interactions, ont, filter_type):
    """ Plots graph of the performance of the model (F1-max) as a function
    of the number of training samples for given term

    :param term_results_file: file path of the term centric results file
    :type term_results_file: str
    :param skip_rows: how many row to skip from the beginning of the file
    :type skip_rows: int
    :param file_test: name of the file for the test data
    :type file_test: str
    :param t1_file: name of the file containing the annotations from time step 1
    :type t1_file: str
    :param file_interactions: name of the file with protein-protein interactions
    :type file_interactions: str
    :type ont: str
    :param filter_type: current filter type of the protein interactions (700 or 900)
    :type filter_type: str
    :return: None
    """
    test_data, graph = prepare_test_data(file_interactions, file_test, ont, filter_type)

    t1_ann = pd.read_csv(t1_file, header=0, sep='\t',
                         names=['PID', 'GO']).groupby('PID')['GO'].apply(list).to_dict()
    anno = set([x for y in list(t1_ann.values()) for x in y])
    mlb = MultiLabelBinarizer()
    mlb.fit([anno])

    matrix = []
    for vertex in graph.vs:
        anno = set(t1_ann[vertex['name']]) if vertex['name'] in t1_ann else set()
        annotations = mlb.transform([anno])[0]
        matrix.append(annotations)
    t1_anno = np.array(matrix)

    count = 0
    x = []
    y = []
    aucs = []
    with open(term_results_file, 'r') as f:
        for line in f.readlines():
            count += 1
            if count <= skip_rows:
                continue
            if line == '\n':
                continue
            line = line.strip()
            parts = line.split(',')
            term = np.argmax(mlb.transform([{parts[0]}])[0])
            freq = np.sum(t1_anno[:, term])
            x.append(freq)
            y.append(float(parts[1]))
            aucs.append(float(parts[2]))

    plot_term_centric_performance_graph(x, y, filter_type, ont)
    plot_aucs(aucs, filter_type, ont)


if __name__ == '__main__':
    filtering_type = '700'
    onto = 'MF'
    # file = f'data/final/human_ppi_{filtering_type}/HumanPPI_{onto}_no_bias.txt'
    # plot_ppi_network_stats(file, filtering_type)
    # plot_annotations_num_levels(onto, filtering_type)
    interactions_file = f'data/final/human_ppi_{filtering_type}/HumanPPI_{onto}_no_bias.txt'
    test_file = f'data/trained/human_ppi_{filtering_type}/test_data_{onto}.pkl'
    term_centric_file = f'data/trained/human_ppi_{filtering_type}/term_centric_evaluation_results_anno_srw_{onto}.csv'
    t1_annotations = f'data/human_ppi_{filtering_type}/HumanPPI_GO_{onto}_no_bias.txt'
    plot_term_centric_f_measure(term_centric_file, 17403, test_file,
                                t1_annotations, interactions_file, onto, filtering_type)
