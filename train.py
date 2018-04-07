import numpy as np
from igraph import *
import pandas as pd
from supervised_random_walks import supervised_random_walks
import time


graph_data = pd.read_csv('toy_graph.csv', header=0)
graph = Graph(directed=True)
vertices = np.unique(np.concatenate((graph_data[['from']].values, graph_data[['to']].values))).tolist()
graph.add_vertices(vertices)
graph.add_edges(graph_data[['from', 'to']].values)
'''
for feature in graph_data.columns[2:]:
    m = np.mean(graph_data[feature].values)
    stddev = np.std(graph_data[feature].values)
    graph.es[feature] = ((graph_data[feature].values - m) / stddev).tolist()
'''

graph.es['prop1'] = [-0.8400269, 0.5040161, -0.8400269, 0.5040161, 1.8480591, -0.8400269, 0.5040161, -0.8400269]
graph.es['prop2'] = [-0.3535534, 1.0606602, 1.0606602, -0.3535534, -0.3535534, 1.0606602, -1.7677670, -0.3535534]

graph.es['const'] = np.ones(graph_data.shape[0]).tolist()

print(graph.es[0].attributes())
start = time.time()
w = supervised_random_walks(graph, [graph.vs[0].index], [graph.vs[1, 3].indices])
print(w)
print(time.time() - start)
