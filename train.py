import time
import pickle
import numpy as np
from igraph import *
import pandas as pd
from supervised_random_walks import supervised_random_walks as srw, random_walks as rw
from supervised_random_walks_gpu import supervised_random_walks as srw_gpu, random_walks as rw_gpu


def train_dummy_example():
    graph_data = pd.read_csv('toy_graph.csv', header=0)
    graph = Graph(directed=True)
    vertices = np.unique(np.concatenate((graph_data[['from']].values, graph_data[['to']].values))).tolist()
    graph.add_vertices(vertices)
    graph.add_edges(graph_data[['from', 'to']].values)

    for feature in graph_data.columns[2:]:
        m = np.mean(graph_data[feature].values)
        stddev = np.std(graph_data[feature].values)
        graph.es[feature] = ((graph_data[feature].values - m) / stddev).tolist()

    graph.es['const'] = np.ones(graph_data.shape[0]).tolist()

    print(graph.es[0].attributes())
    start = time.time()
    result = srw(graph.copy(), [graph.vs[0].index], [graph.vs[1, 3].indices])
    if result[2]['warnflag'] == 0:
        print('Optimization converged!')
    else:
        print('No convergence. ' + str(result[2]['task']))
    w = result[0]
    print('Optimized parameters: ')
    print(w)
    print('Training elapsed time: ' + str(time.time() - start))

    start = time.time()
    result = srw_gpu(graph.copy(), [graph.vs[0].index], [graph.vs[1, 3].indices])
    if result[2]['warnflag'] == 0:
        print('Optimization converged!')
    else:
        print('No convergence. ' + str(result[2]['task']))
    w = result[0]
    print('Optimized parameters: ')
    print(w)
    print('Training elapsed time: ' + str(time.time() - start))

    print('\nTesting')
    p_vectors = rw(graph.copy(), w, graph.vs.indices)
    print(p_vectors)


def train_protein_based_function_prediction(graph, sources, destinations, gpu=False):
    start = time.time()
    if gpu:
        result = srw_gpu(graph.copy(), sources, destinations)
    else:
        result = srw(graph.copy(), sources, destinations)
    if result[2]['warnflag'] == 0:
        print('Optimization converged!')
    else:
        print('No convergence. ' + str(result[2]['task']))
    w = result[0]
    print('Optimized parameters: ')
    print(w)
    print('Training elapsed time: ' + str(time.time() - start))
    return w


def protein_based_function_prediction(file_interactions, file_train, t1_file, t2_file, file_weights, ont, filter_type):
    if os.path.exists(file_interactions):
        ppi = pd.read_csv(file_interactions, header=0, sep='\t').fillna(0)
        scores = ppi.combined_score.values
        ppi.combined_score = (scores - int(filter_type)) / (1000 - int(filter_type))
    else:
        raise Exception('{} does not exist'.format(file_interactions))

    # graph = Graph(directed=True)
    graph = Graph(directed=False)
    vertices = np.unique(np.concatenate((ppi[['protein1']].values, ppi[['protein2']].values))).tolist()
    graph.add_vertices(vertices)
    graph.add_edges(ppi[['protein1', 'protein2']].values)
    for feature in ppi.columns[9:]:
        graph.es[feature] = ppi[feature].values.tolist()

    del ppi

    indices = graph.vs['name']
    sources = [indices.index(source)
               for source in pd.read_csv(file_train, header=0, sep='\t').protein_id.values.tolist()
               if source in indices]
    t1_ann = pd.read_csv(t1_file, header=0, sep='\t', names=['PID', 'GO']).groupby('PID')['GO'].apply(list).to_dict()
    t2_ann = pd.read_csv(t2_file, header=0, sep='\t', names=['PID', 'GO']).groupby('PID')['GO'].apply(list).to_dict()
    if os.path.exists(f'data/trained/human_ppi_{filter_type}/train_destinations_{ont}.pkl'):
        with open(f'data/trained/human_ppi_{filter_type}/train_destinations_{ont}.pkl', 'rb') as f:
            destinations = pickle.load(f)
    else:
        destinations = []
        for source in sources:
            anno = set(t2_ann[graph.vs[source]['name']])
            dests = []
            for vertex in graph.vs:
                if vertex.index == source:
                    continue
                if vertex['name'] in t1_ann and len(anno.intersection(t1_ann[vertex['name']])) > 0:
                    dests.append(vertex.index)
            destinations.append(dests)

        with open(f'data/trained/human_ppi_{filter_type}/train_destinations_{ont}.pkl', 'wb') as f:
            pickle.dump(destinations, f, pickle.HIGHEST_PROTOCOL)

    w = train_protein_based_function_prediction(graph, sources[:3], destinations[:3])
    # w = train_protein_based_function_prediction(graph, sources, destinations, gpu=True)
    np.savetxt(file_weights, w)


if __name__ == '__main__':
    # train_dummy_example()
    filtering_type = '700'
    ont = 'MF'
    file = f'data/final/human_ppi_{filtering_type}/HumanPPI_{ont}_no_bias.txt'
    train_file = f'data/final/human_ppi_{filtering_type}/train_{ont}_no_bias.txt'
    t1_annotations = f'data/human_ppi_{filtering_type}/HumanPPI_GO_{ont}_no_bias.txt'
    t2_annotations = f'data/human_ppi_{filtering_type}/t2/HumanPPI_GO_{ont}_no_bias.txt'
    weights_file = f'data/trained/human_ppi_{filtering_type}/weights_{ont}_no_bias.txt'
    protein_based_function_prediction(file, train_file, t1_annotations, t2_annotations,
                                      weights_file, ont, filtering_type)
