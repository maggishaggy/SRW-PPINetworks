import time
import numpy as np
from igraph import *
import pandas as pd
# from supervised_random_walks import supervised_random_walks, random_walks
from supervised_random_walks_gpu import supervised_random_walks, random_walks


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
    result = supervised_random_walks(graph.copy(), [graph.vs[0].index], [graph.vs[1, 3].indices])
    if result[2]['warnflag'] == 0:
        print('Optimization converged!')
    else:
        print('No convergence. ' + result[2]['task'])
    w = result[0]
    print('Optimized parameters: ')
    print(w)
    print('Training elapsed time: ' + str(time.time() - start))

    print('\nTesting')
    p_vectors = random_walks(graph.copy(), w, graph.vs.indices)
    print(p_vectors)


def train_protein_based_function_prediction(graph, sources, destnations):
    start = time.time()
    result = supervised_random_walks(graph.copy(), sources, destnations)
    if result[2]['warnflag'] == 0:
        print('Optimization converged!')
    else:
        print('No convergence. ' + result[2]['task'])
    w = result[0]
    print('Optimized parameters: ')
    print(w)
    print('Training elapsed time: ' + str(time.time() - start))
    return w


def protein_based_function_prediction(file_path):
    if os.path.exists(file_path):
        ppi = pd.read_csv(file_path, header=0)[:1000]
    else:
        raise Exception('{} does not exist'.format(file_path))

    graph = Graph(directed=True)
    vertices = np.unique(np.concatenate((ppi[['protein1']].values, ppi[['protein2']].values))).tolist()
    graph.add_vertices(vertices)
    graph.add_edges(ppi[['protein1', 'protein2']].values)
    graph.add_edges(ppi[['protein2', 'protein1']].values)
    for feature in ppi.columns[2:]:
        graph.es[feature] = ppi[feature].values.tolist() + ppi[feature].values.tolist()

    sources = np.random.choice(np.array(graph.vs.indices), 1).tolist()
    destinations = []
    for source in sources:
        possibilities = np.array(list(set(graph.vs.indices).difference([source])))
        destination = np.random.choice(possibilities, 50).tolist()
        destinations.append(destination)

    w = train_protein_based_function_prediction(graph, sources, destinations)


if __name__ == '__main__':
    # train_dummy_example()
    file = 'data/HumanPPI700_interactions_BP_normalized.txt'
    protein_based_function_prediction(file)
