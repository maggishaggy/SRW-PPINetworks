import pickle
import numpy as np
import pandas as pd
from igraph import *
import tensorflow as tf
from config import Config
from anno_srw_model import AnnoSRW
from sklearn.preprocessing import MultiLabelBinarizer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def train_model(file_interactions, file_train, t1_file, t2_file, ont, filter_type):

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

    if os.path.exists(f'data/trained/human_ppi_{filter_type}/train_data_{ont}.pkl'):
        with open(f'data/trained/human_ppi_{filter_type}/train_data_{ont}.pkl', 'rb') as f:
            train_data = pickle.load(f)
    else:
        train_data = dict()
        train_data['features'] = np.array([graph.es[feature] for feature in graph.es.attributes()]).T
        train_data['adj'] = np.array(graph.get_adjacency().data)
        train_data['data'] = []

        indices = graph.vs['name']
        sources = [indices.index(source)
                   for source in pd.read_csv(file_train, header=0, sep='\t').protein_id.values.tolist()
                   if source in indices]
        train_data['num_sources'] = len(sources)
        t1_ann = pd.read_csv(t1_file, header=0, sep='\t',
                             names=['PID', 'GO']).groupby('PID')['GO'].apply(list).to_dict()
        anno = set([x for y in list(t1_ann.values()) for x in y])
        t2 = pd.read_csv(t2_file, header=0, sep='\t', names=['PID', 'GO'])
        t2 = t2[t2.GO.isin(anno)]
        t2_ann = t2.groupby('PID')['GO'].apply(list).to_dict()
        mlb = MultiLabelBinarizer()
        mlb.fit([anno])

        train_data['num_classes'] = len(anno)

        matrix = []
        for vertex in graph.vs:
            anno = set(t1_ann[vertex['name']]) if vertex['name'] in t1_ann else set()
            annotations = mlb.transform([anno])[0]
            matrix.append(annotations)
        train_data['t1_anno'] = np.array(matrix)

        for source in sources:
            anno = set(t2_ann[graph.vs[source]['name']])
            annotations = mlb.transform([anno])[0]
            train_data['data'].append({'source': source, 'annotations': annotations})

        with open(f'data/trained/human_ppi_{filter_type}/train_data_{ont}.pkl', 'wb') as f:
            pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

    conf = Config(num_vertices=len(graph.vs.indices), num_features=7, alpha=0.3,
                  lambda_param=1, margin_loss=0.4, max_iter=500, epsilon=1e-12,
                  small_epsilon=1e-18, summary_dir=f'summary_anno_{ont}', save_dir=f'models_anno_{ont}',
                  num_classes=train_data['num_classes'])
    with tf.Session() as sess:
        model = AnnoSRW(conf, mode='training')
        sess.run(tf.global_variables_initializer())
        # model.load(sess)
        tf.get_default_graph().finalize()
        model.train(sess, train_data)


if __name__ == '__main__':
    filtering_type = '700'
    onto = 'MF'
    file = f'data/final/human_ppi_{filtering_type}/HumanPPI_{onto}_no_bias.txt'
    train_file = f'data/final/human_ppi_{filtering_type}/train_{onto}_no_bias.txt'
    t1_annotations = f'data/human_ppi_{filtering_type}/HumanPPI_GO_{onto}_no_bias.txt'
    t2_annotations = f'data/human_ppi_{filtering_type}/t2/HumanPPI_GO_{onto}_no_bias.txt'
    train_model(file, train_file, t1_annotations, t2_annotations,
                onto, filtering_type)
