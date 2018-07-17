import pickle
import numpy as np
import pandas as pd
from igraph import *
from tqdm import tqdm
import tensorflow as tf
from config import Config
from srw_model import SRW
from sklearn.preprocessing import MultiLabelBinarizer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def prepare_test_data(file_interactions, file_test, ont, filter_type):
    """ Prepares the test data and protein interactions graph

    :param file_interactions: name of the file with protein-protein interactions
    :type file_interactions: str
    :param file_test: name of the file for the test data
    :type file_test: str
    :param ont: name of the current ontology (BP, CC or MF)
    :type ont: str
    :param filter_type: current filter type of the protein interactions (700 or 900)
    :type filter_type: str
    :return: test data dictionary, graph
    :rtype: dict, igraph Graph object
    """
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

    return test_data, graph


def find_num_mutual(rel, ret):
    return len(set(rel).intersection(set(ret)))


def calculate_measures(rel, ret):
    """ Calculates Precision, Recall, and F1 measures

    :param rel: list of ground truth GO terms
    :type rel: list
    :param ret: list of predicted GO terms
    :type ret: list
    :return:
    """
    intersect = find_num_mutual(rel, ret)
    precision = intersect * 1.0 / len(ret) if len(ret) > 0 else 0
    recall = intersect * 1.0 / len(rel) if len(rel) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return round(precision, 10), round(recall, 10), round(f1, 10)


def eval_predictions_protein_centric(source, scores, t2_anno, mlb, t):
    """ Calculates protein-centric metrics for predictions

    :param source: STRING ID of the source protein
    :type source: str
    :param scores: prediction scores
    :type scores: list(numpy.array)
    :param t2_anno: annotations from t2 step
    :type t2_anno: dict
    :param mlb: MultiLabelBinarizer trained model
    :type: sklearn.preprocessing.MultiLabelBinarizer object
    :param t: threshold for predictions
    :type t: float
    :return:
    """
    rel_anno = t2_anno[source]
    ret_anno = list(mlb.inverse_transform(np.array([np.where(scores[0] > t, 1, 0)]))[0])
    pr, re, f1 = calculate_measures(rel_anno, ret_anno)
    return pr, re, f1, len(ret_anno)


def eval_predictions_term_centric(term, scores, t2_anno, mlb, indices, t):
    """ Calculates term-centric metrics for predictions

    :param term: GO term
    :type term: str
    :param scores: prediction scores
    :type scores: list(numpy.array)
    :param t2_anno: annotations from t2 step
    :type t2_anno: dict
    :param mlb: MultiLabelBinarizer trained model
    :type: sklearn.preprocessing.MultiLabelBinarizer object
    :param indices: STRING ID of the source proteins according to the index
    :type indices: dict
    :param t: threshold for predictions
    :type t: float
    :return:
    """
    rel_ret = 0
    rel = 0
    ret = 0
    not_rel_not_ret = 0
    not_rel = 0
    for p in scores.keys():
        rel_anno = t2_anno[indices[p]]
        ret_anno = list(mlb.inverse_transform(np.array([np.where(scores[p][0] > t, 1, 0)]))[0])
        if term in rel_anno and term in ret_anno:
            rel_ret += 1
        if term in ret_anno:
            ret += 1
        if term in rel_anno:
            rel += 1
        else:
            not_rel += 1
        if term not in rel_anno and term not in ret_anno:
            not_rel_not_ret += 1
    se = rel_ret * 1.0 / rel if rel != 0 else 0
    sp = not_rel_not_ret * 1.0 / not_rel if not_rel != 0 else 0
    re = se
    pr = rel_ret * 1.0 / ret if ret != 0 else 0
    f1 = (2 * pr * re) / (pr + re) if pr + re != 0 else 0
    return round(se, 10), round(sp, 10), round(re, 10), round(pr, 10), round(f1, 10)


def f1_measure(avg_p, avg_r):
    if (avg_p + avg_r) == 0:
        return 0
    return round((2 * avg_p * avg_r) / (avg_p + avg_r), 10)


def maximum_f1_measure(array):
    return round(np.max(array), 10)


def average_precision(array_1, array_2):
    if np.count_nonzero(array_2) == 0:
        return 0.0
    return round(np.sum(array_1) / np.count_nonzero(array_2), 10)


def average_recall(array):
    return round(np.average(array), 10)


def create_test_terms(protein_sources, t2_anno, indices):
    terms = [t2_anno[indices[source]] for source in protein_sources if indices[source] in t2_anno.keys()]
    return np.unique(np.concatenate(terms))


def write_results_protein_centric(method, ont, filter_type, measures, max_f1):
    with open(f'data/trained/human_ppi_{filter_type}/protein_centric_evaluation_results_{method}_{ont}.csv', 'w+') as f:
        f.write('Threshold,PID,Precision,Recall,F1-measure\n')
        for t in measures.keys():
            for key in measures[t].keys():
                value = measures[t][key]
                if type(value) is list:
                    f.write(f'{t},{key},{value[0]},{value[1]},{value[2]}\n')
        f.write('\n')
        f.write('Threshold,Average precision\n')
        for t in measures.keys():
            f.write(f'{t},{measures[t]["Average precision"]}\n')
        f.write('\n')
        f.write('Threshold,Average recall\n')
        for t in measures.keys():
            f.write(f'{t},{measures[t]["Average recall"]}\n')
        f.write('\n')
        f.write('Threshold,F1-measure\n')
        for t in measures.keys():
            f.write(f'{t},{measures[t]["F1-measure"]}\n')
        f.write('\n')
        f.write(f'Maximum F1-measure,{max_f1}\n')


def write_results_term_centric(method, ont, filter_type, measures, max_f1_measures):
    with open(f'data/trained/human_ppi_{filter_type}/term_centric_evaluation_results_{method}_{ont}.csv', 'w+') as f:
        f.write('Threshold,GO ID,Sensitivity,Specificity,Recall,Precision,F1-measure\n')
        for t in measures.keys():
            for key in measures[t].keys():
                value = measures[t][key]
                f.write(f'{t},{key},{value[0]},{value[1]},{value[2]},{value[3]},{value[4]}\n')
        f.write('\n')
        f.write(f'GO ID,Maximum F1-measure\n')
        for key in max_f1_measures.keys():
            f.write(f'{key},{max_f1_measures[key]}\n')
        f.write('\n')


def evaluate(method, test_data, graph, results, t1_file, t2_file, ont, filter_type, thresholds):
    """ Make evaluation of the predictions with protein- and term-centric metrics

    :param method: name of the method used to make the predictions
    :type method: str
    :param test_data: test data
    :type test_data: dict
    :param graph: protein interactions graph
    :type graph: igraph Graph object
    :param results: predictions of the method, dictionary with key protein id and value p vector
    :type results: dict
    :param t1_file: name of the file containing the annotations from time step 1
    :type t1_file: str
    :param t2_file: name of the file containing the annotations from time step 2
    :type t2_file: str
    :param ont: name of the current ontology (BP, CC or MF)
    :type ont: str
    :param filter_type: current filter type of the protein interactions (700 or 900)
    :type filter_type: str
    :param thresholds: thresholds values for probabilities of predictions
    :type thresholds: list(float)
    :return: None
    """
    t1_ann = pd.read_csv(t1_file, header=0, sep='\t', names=['PID', 'GO']).groupby('PID')['GO'].apply(list).to_dict()
    anno = set([x for y in list(t1_ann.values()) for x in y])
    t2 = pd.read_csv(t2_file, header=0, sep='\t', names=['PID', 'GO'])
    t2 = t2[t2.GO.isin(anno)]
    t2_ann = t2.groupby('PID')['GO'].apply(list).to_dict()
    mlb = MultiLabelBinarizer()
    mlb.fit([anno])

    indices = graph.vs['name']

    test_terms = create_test_terms(test_data['sources'], t2_ann, indices)
    measures = dict()
    max_f1_measures = dict()
    for t in tqdm(thresholds):
        measures[t] = dict()
        for term in tqdm(test_terms):
            se, sp, re, pr, f1 = eval_predictions_term_centric(term, results, t2_ann, mlb, indices, t)
            measures[t][term] = [se, sp, re, pr, f1]
            if term not in max_f1_measures.keys() or max_f1_measures[term] < f1:
                max_f1_measures[term] = f1
    write_results_term_centric(method, ont, filter_type, measures, max_f1_measures)

    measures = dict()
    f1_measures = []
    for t in tqdm(thresholds):
        precision, recall, number_of_predictions = [], [], []
        measures[t] = dict()
        for source in tqdm(test_data['sources']):
            pr, re, f1, num = eval_predictions_protein_centric(indices[source], results[source], t2_ann, mlb, t)
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
    write_results_protein_centric(method, ont, filter_type, measures, max_f1)


def calc_probability_classes(t1_file, results, graph):
    """ Calculates the class probabilities from the p vector

    :param t1_file: name of the file containing the annotations from time step 1
    :type t1_file: str
    :param results: p vector for each source node
    :type results: dict
    :param graph: protein interactions graph
    :type graph: igraph Graph object
    :return: class probabilities
    :rtype: dict
    """
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

    for key, value in results.items():
        results[key] = [np.matmul(value[0].reshape([1, -1]), t1_anno).reshape(-1)]
    return results


def test_model(file_interactions, file_test, t1_file, t2_file,
               ont, filter_type, model_file, thresholds):
    """ Evaluate the Supervised Random Walks algorithm with protein- and term-centric metrics

    :param file_interactions: name of the file with protein-protein interactions
    :type file_interactions: str
    :param file_test: name of the file for the test data
    :type file_test: str
    :param t1_file: name of the file containing the annotations from time step 1
    :type t1_file: str
    :param t2_file: name of the file containing the annotations from time step 2
    :type t2_file: str
    :param ont: name of the current ontology (BP, CC or MF)
    :type ont: str
    :param filter_type: current filter type of the protein interactions (700 or 900)
    :type filter_type: str
    :param model_file: name of the file containing the trained weights
    :type model_file: str
    :param thresholds: thresholds values for probabilities of predictions
    :type thresholds: list(float)
    :return: None
    """
    test_data, graph = prepare_test_data(file_interactions, file_test, ont, filter_type)

    if os.path.exists(f'data/trained/human_ppi_{filter_type}/predictions_srw{ont}.pkl'):
        with open(f'data/trained/human_ppi_{filter_type}/predictions_srw{ont}.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        conf = Config(num_vertices=len(graph.vs.indices), num_features=7, alpha=0.3,
                      lambda_param=1, margin_loss=0.4, max_iter=10000, epsilon=1e-12,
                      small_epsilon=1e-18, summary_dir=f'summary_{ont}', save_dir=f'models_{ont}')
        with tf.Session() as sess:
            model = SRW(conf, mode='inference')
            sess.run(tf.global_variables_initializer())
            model.load(sess, model_file)
            # weights = tf.global_variables()[0]
            # sess.run(weights.assign(np.array([-7.565122782132602, 0.43171067626253357, 1.3190980108033072,
            #                                   -3.42624258727104, 1.0739606936051764, 6.334479697770692,
            #                                   4.419444407845334]).reshape(-1, 1)))
            tf.get_default_graph().finalize()
            results = model.predict(sess, test_data)

        results = calc_probability_classes(t1_file, results, graph)

        with open(f'data/trained/human_ppi_{filter_type}/predictions_srw{ont}.pkl', 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    evaluate('srw', test_data, graph, results, t1_file, t2_file, ont, filter_type, thresholds)


def test_swr_no_weights(file_interactions, file_test, t1_file, t2_file,
                        ont, filter_type, thresholds):
    """ Evaluate the Supervised Random Walks algorithm (no learned weights) with protein- and term-centric metrics

    :param file_interactions: name of the file with protein-protein interactions
    :type file_interactions: str
    :param file_test: name of the file for the test data
    :type file_test: str
    :param t1_file: name of the file containing the annotations from time step 1
    :type t1_file: str
    :param t2_file: name of the file containing the annotations from time step 2
    :type t2_file: str
    :param ont: name of the current ontology (BP, CC or MF)
    :type ont: str
    :param filter_type: current filter type of the protein interactions (700 or 900)
    :type filter_type: str
    :param thresholds: thresholds values for probabilities of predictions
    :type thresholds: list(float)
    :return: None
    """
    test_data, graph = prepare_test_data(file_interactions, file_test, ont, filter_type)
    if os.path.exists(f'data/trained/human_ppi_{filter_type}/predictions_srw_no_weights{ont}.pkl'):
        with open(f'data/trained/human_ppi_{filter_type}/predictions_srw_no_weights{ont}.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        conf = Config(num_vertices=len(graph.vs.indices), num_features=7, alpha=0.3,
                      lambda_param=1, margin_loss=0.4, max_iter=10000, epsilon=1e-12,
                      small_epsilon=1e-18, summary_dir=f'summary_{ont}', save_dir=f'models_{ont}')

        with tf.Session() as sess:
            model = SRW(conf, mode='inference')
            weights = tf.global_variables()[0]
            sess.run(weights.assign(np.ones(tuple(weights.shape.as_list()))))
            tf.get_default_graph().finalize()
            results = model.predict(sess, test_data)

        results = calc_probability_classes(t1_file, results, graph)

        with open(f'data/trained/human_ppi_{filter_type}/predictions_srw_no_weights{ont}.pkl', 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    evaluate('srw_no_weights', test_data, graph, results, t1_file, t2_file, ont, filter_type, thresholds)


def test_random_walks(file_interactions, file_test, t1_file, t2_file,
                      ont, filter_type, thresholds):
    """ Evaluate the Random Walks algorithm with protein- and term-centric metrics

    :param file_interactions: name of the file with protein-protein interactions
    :type file_interactions: str
    :param file_test: name of the file for the test data
    :type file_test: str
    :param t1_file: name of the file containing the annotations from time step 1
    :type t1_file: str
    :param t2_file: name of the file containing the annotations from time step 2
    :type t2_file: str
    :param ont: name of the current ontology (BP, CC or MF)
    :type ont: str
    :param filter_type: current filter type of the protein interactions (700 or 900)
    :type filter_type: str
    :param thresholds: thresholds values for probabilities of predictions
    :type thresholds: list(float)
    :return: None
    """
    test_data, graph = prepare_test_data(file_interactions, file_test, ont, filter_type)
    test_data['features'] = np.ones((test_data['features'].shape[0], 1))
    if os.path.exists(f'data/trained/human_ppi_{filter_type}/predictions_random_walk{ont}.pkl'):
        with open(f'data/trained/human_ppi_{filter_type}/predictions_random_walk{ont}.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        conf = Config(num_vertices=len(graph.vs.indices), num_features=1, alpha=0.3,
                      lambda_param=1, margin_loss=0.4, max_iter=10000, epsilon=1e-12,
                      small_epsilon=1e-18, summary_dir=f'summary_{ont}', save_dir=f'models_{ont}')

        with tf.Session() as sess:
            model = SRW(conf, mode='inference')
            weights = tf.global_variables()[0]
            sess.run(weights.assign(np.ones(tuple(weights.shape.as_list()))))
            tf.get_default_graph().finalize()
            results = model.predict(sess, test_data)

        results = calc_probability_classes(t1_file, results, graph)

        with open(f'data/trained/human_ppi_{filter_type}/predictions_random_walk{ont}.pkl', 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    evaluate('random_walk', test_data, graph, results, t1_file, t2_file, ont, filter_type, thresholds)


if __name__ == '__main__':
    filtering_type = '700'
    onto = 'CC'
    file = f'data/final/human_ppi_{filtering_type}/HumanPPI_{onto}_no_bias.txt'
    test_file = f'data/final/human_ppi_{filtering_type}/test_{onto}_no_bias.txt'
    t1_annotations = f'data/human_ppi_{filtering_type}/HumanPPI_GO_{onto}_no_bias.txt'
    t2_annotations = f'data/human_ppi_{filtering_type}/t2/HumanPPI_GO_{onto}_no_bias.txt'
    trained_model_file = f'data/trained/human_ppi_{filtering_type}/model_{onto}_no_bias.npy'
    test_model(file, test_file, t1_annotations, t2_annotations,
               onto, filtering_type, trained_model_file, np.arange(0.0, 1.0, 0.01).tolist())
    # test_swr_no_weights(file, test_file, t1_annotations, t2_annotations,
    #                     onto, filtering_type, np.arange(0.0, 1.0, 0.01).tolist())
    # test_random_walks(file, test_file, t1_annotations, t2_annotations,
    #                   onto, filtering_type, np.arange(0.0, 1.0, 0.01).tolist())
