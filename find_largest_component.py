import pandas as pd
import networkx as nx


def extract_component(file_name):
    x = pd.read_table(file_name, sep=',')
    g = nx.from_pandas_dataframe(x, 'protein1', 'protein2', True)
    largest = max(nx.connected_component_subgraphs(g), key=len)
    for i in range(x.shape[0]):
        if x.at[i, 'protein1'] not in largest.nodes() and x.at[i, 'protein2'] not in largest.nodes():
            x.at[i, 'protein1'] = 'x'
    x = x[x['protein1'] != 'x']
    x.to_csv(file_name, sep=',', index=False)


HumanPPI700 = 'Data/HumanPPI700_interactions.txt'
HumanPPI900 = 'Data/HumanPPI900_interactions.txt'
extract_component(HumanPPI700)
extract_component(HumanPPI900)
