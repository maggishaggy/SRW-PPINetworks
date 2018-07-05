import pickle
import numpy as np
import pandas as pd
from igraph import *
import tensorflow as tf
from config import Config
from srw_model import SRW

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def find_num_mutual(rel, ret):
    return len(set(rel).intersection(set(ret)))


def calculate_measures(rel, ret):
    intersect = find_num_mutual(rel, ret)
    precision = intersect * 1.0 / len(ret) if len(ret) > 0 else 0
    recall = intersect * 1.0 / len(rel) if len(rel) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return round(precision, 10), round(recall, 10), round(f1, 10)


def eval_predictions_protein_centric(source, scores, t1_anno, t2_anno, indices, n, t=None):
    preds = sorted([(i, s) for i, s in zip(range(len(scores[0])), scores[0])], key=lambda x: x[1], reverse=True)[:n]
    if t is not None:
        preds = [p for p in preds if p[1] > t]
    rel_anno = t2_anno[indices[source]]
    ret_anno = [t1_anno[indices[p[0]]] for p in preds if indices[p[0]] in t1_anno.keys()]
    ret_anno = np.unique(np.concatenate(ret_anno)) if len(ret_anno) > 0 else []
    pr, re, f1 = calculate_measures(rel_anno, ret_anno)
    return pr, re, f1, len(ret_anno)


def f1_measure(avg_p, avg_r):
    return round((2 * avg_p * avg_r) / (avg_p + avg_r), 10)


def maximum_f1_measure(array):
    return round(np.max(array), 10)


def average_precision(array_1, array_2):
    return round(np.sum(array_1) / np.count_nonzero(array_2), 10)


def average_recall(array):
    return round(np.average(array), 10)


def write_results(ont, filter_type, measures, max_f1, n):
    with open(f'data/trained/human_ppi_{filter_type}/evaluation_results_{ont}.csv', 'w+') as f:
        f.write('Threshold,PID,Precision,Recall,F1-measure\n')
        for t in measures.keys():
            for key in measures[t].keys():
                value = measures[t][key]
                if type(value) is list:
                    f.write(f'{t},{key},{value[0]},{value[1]},{value[2]}\n')
                else:
                    f.write(f'{key},{value}\n')
        f.write(f'Maximum F1-measure,{max_f1}\n')
        f.write(f'Number of proteins for prediction,{n}\n')


def test_model(file_interactions, file_test, t1_file, t2_file, ont, filter_type, model_file, n, thresholds):

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

    if os.path.exists(f'data/trained/human_ppi_{filter_type}/test_data_{ont}.pkl'):
        with open(f'data/trained/human_ppi_{filter_type}/test_data_{ont}.pkl', 'rb') as f:
            test_data = pickle.load(f)
    else:
        test_data = dict()
        test_data['features'] = np.array([graph.es[feature] for feature in graph.es.attributes()]).T
        test_data['adj'] = np.array(graph.get_adjacency().data)

        indices = graph.vs['name']
        sources = [indices.index(source)
                   for source in pd.read_csv(file_test, header=0, sep='\t').protein_id.values.tolist()
                   if source in indices]
        test_data['num_sources'] = len(sources)
        test_data['sources'] = sources

        with open(f'data/trained/human_ppi_{filter_type}/test_data_{ont}.pkl', 'wb') as f:
            pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

    if os.path.exists(f'data/trained/human_ppi_{filter_type}/predictions_{ont}.pkl'):
        with open(f'data/trained/human_ppi_{filter_type}/predictions_{ont}.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        conf = Config(num_vertices=len(graph.vs.indices), num_features=7, alpha=0.3,
                      lambda_param=1, margin_loss=0.4, max_iter=100, epsilon=1e-12,
                      small_epsilon=1e-18, summary_dir=f'summary_{ont}', save_dir=f'models_{ont}')
        with tf.Session() as sess:
            model = SRW(conf, mode='inference')
            sess.run(tf.global_variables_initializer())
            model.load(sess, model_file)
            tf.get_default_graph().finalize()
            results = model.predict(sess, test_data)
        with open(f'data/trained/human_ppi_{filter_type}/predictions_{ont}.pkl', 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    t1_ann = pd.read_csv(t1_file, header=0, sep='\t', names=['PID', 'GO']).groupby('PID')['GO'].apply(list).to_dict()
    t2_ann = pd.read_csv(t2_file, header=0, sep='\t', names=['PID', 'GO']).groupby('PID')['GO'].apply(list).to_dict()
    indices = graph.vs['name']
    measures = dict()
    f1_measures = []
    for t in thresholds:
        precision, recall, number_of_predictions = [], [], []
        measures[t] = dict()
        for source in test_data['sources']:
            pr, re, f1, num = eval_predictions_protein_centric(source, results[source], t1_ann, t2_ann, indices, n, t)
            measures[t][indices[source]] = [pr, re, f1]
            precision.append(pr)
            recall.append(re)
            number_of_predictions.append(num)
        avg_p = average_precision(precision, number_of_predictions)
        avg_r = average_recall(recall)
        f_1 = f1_measure(avg_p, avg_r)
        f1_measures.append(f_1)
        measures[t]['Average precision'] = avg_p
        measures[t]['Average recall'] = avg_r
        measures[t]['F1-measure'] = f_1
    max_f1 = maximum_f1_measure(f1_measures)
    write_results(ont, filter_type, measures, max_f1, n)


if __name__ == '__main__':
    filtering_type = '700'
    onto = 'CC'
    file = f'data/final/human_ppi_{filtering_type}/HumanPPI_{onto}_no_bias.txt'
    test_file = f'data/final/human_ppi_{filtering_type}/test_{onto}_no_bias.txt'
    t1_annotations = f'data/human_ppi_{filtering_type}/HumanPPI_GO_{onto}_no_bias.txt'
    t2_annotations = f'data/human_ppi_{filtering_type}/t2/HumanPPI_GO_{onto}_no_bias.txt'
    model_file = f'data/trained/human_ppi_{filtering_type}/model_{onto}_no_bias.npy'
    test_model(file, test_file, t1_annotations, t2_annotations,
               onto, filtering_type, model_file, 5, [0.05, 0.1, 0.3])
