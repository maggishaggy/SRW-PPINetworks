from igraph import *
import pandas as pd
import numpy as np
from supervised_random_walks import supervised_random_walks


graph_data = pd.read_csv('toy_graph.csv', header=0)
graph = Graph()
vertices = np.unique(np.concatenate((graph_data[['from']].values, graph_data[['to']].values))).tolist()
graph.add_vertices(vertices)
graph.add_edges(graph_data[['from', 'to']].values)
for feature in graph_data.columns[2:]:
    graph.es[feature] = graph_data[feature].values.tolist()

print(graph.es[0].attributes())
supervised_random_walks(graph, [graph.vs[0].index], [graph.vs[1, 2].indices])

