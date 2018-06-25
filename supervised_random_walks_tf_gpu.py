import os
import time
import functools
import numpy as np
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def logistic_function(x, b):
    """
    Logistic function - Wilcoxon-Mann-Whitney (WMW) loss
    """
    return 1.0 / (1 + tf.exp(-1 * x / b))


def derivative_logistic_function(x, b):
    """
    Derivative of logistic function
    """
    return (logistic_function(x, b) * (1 - logistic_function(x, b)))/b


def get_differences(p, l_set, d_set):
    """ Calculate difference between pl and pd

    :param p: stationary distribution
    :type p: tensorflow.Tensor
    :param l_set: indices of the nodes in L set
    :type l_set: list
    :param d_set: indices of the nodes in D set
    :type d_set: list
    :return: difference between pl and pd
    :rtype: tensorflow.Tensor
    """
    pd = tf.gather(p, d_set)
    pl = tf.gather(p, l_set)
    result = tf.reshape(tf.subtract(tf.expand_dims(pl, 0), tf.expand_dims(pd, 1)), [-1])
    return result


def logistic_edge_strength_function(w, features):
    """ Logistic edge strength function

    :param w: weight vector
    :type w: tensorflow.Tensor
    :param features: feature vector
    :type features: tensorflow.Tensor
    :return: edge strengths
    :rtype: tensorflow.Tensor
    """
    return logistic_function(tf.tensordot(features, w, 1), 1)


def logistic_edge_strength_derivative_function(w, features):
    """ Derivative of logistic edge strength function

    :param w: weight vector
    :type w: tensorflow.Tensor
    :param features: feature vector
    :type features: tensorflow.Tensor
    :return: edge strengths
    :rtype: tensorflow.Tensor
    """
    return tf.multiply(derivative_logistic_function(tf.tensordot(features, w, 1), 1),
                       tf.transpose(features))


def get_stochastic_transition_matrix(A):
    """ Calculates the stochastic transition matrix

    :param A: adjacency matrix with edge strengths
    :type A: tensorflow.Tensor
    :return: stochastic transition matrix
    :rtype: tensorflow.Tensor
    """
    Q_prim = tf.multiply(1 / tf.reduce_sum(A, axis=1, keep_dims=True), A)
    return Q_prim


def get_transition_matrix(Q_prim, start_node, alpha):
    """ Calculate the transition matrix from given stochastic transition matrix,
    start node and restart probability

    :param Q_prim: stochastic transition matrix
    :type Q_prim: tensorflow.Tensor
    :param start_node: index of the start node
    :type start_node: int
    :param alpha: restart probability
    :type alpha: float
    :return: transition matrix
    :rtype: tensorflow.Tensor
    """
    one = np.zeros(Q_prim.shape, dtype=np.float32)
    one[:, start_node] = 1.0
    one = tf.constant(one)
    return tf.add((1 - alpha) * Q_prim, alpha * one)


def loss_function(diff, b):
    """ Loss function

    :param diff: differences pl - pd
    :type diff: tensorflow.Tensor
    :param b: parameter in logistic function
    :type b: float
    :return: loss for each difference
    :rtype: tensorflow.Tensor
    """
    return logistic_function(diff, b)


def iterative_page_rank(trans, epsilon, max_iter):
    """ Iterative power-iterator like computation of PageRank vector p

    :param trans: transition matrix
    :type trans: tensorflow.Tensor
    :param epsilon: tolerance parameter
    :type epsilon: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :return: stationary distribution
    :rtype: tensorflow.Tensor
    """
    p = tf.divide(tf.ones((1, trans.shape[0])), tf.cast(trans.shape[0], tf.float32))
    p_new = tf.tensordot(p, trans, 1)
    with tf.Session():
        while True:
            p = p_new
            p_new = tf.tensordot(p, trans, 1)
            max_iter -= 1
            if tf.reduce_max(tf.abs(tf.subtract(p, p_new))).eval() < epsilon or max_iter <= 0:
                break
    return p_new[0]


def iterative_page_rank_derivative(adj, p, Q, A, epsilon, max_iter, w, features, alpha):
    """ Derivative of PageRank vector p

    :param adj: adjacency matrix
    :type adj: tensorflow.Tensor
    :param p: PageRank vector
    :param p: tensorflow.Tensor
    :param Q: transition matrix
    :type Q: tensorflow.Tensor
    :param A: adjacency strength matrix
    :type A: tensorflow.Tensor
    :param epsilon: tolerance parameter
    :type epsilon: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :param w: parameter vector
    :type w: tensorflow.Tensor
    :param features: feature vector
    :type features: tensorflow.Tensor
    :param alpha: restart probability
    :type alpha: float
    :return: derivative of PageRank vector
    :rtype: tensorflow.Tensor
    """
    dp = tf.zeros((Q.shape[0], w.shape[0]))
    dstrengths = logistic_edge_strength_derivative_function(w, features)
    with tf.Session():
        for k in range(w.shape[0]):
            t = 0
            k_dstrengths = tf.transpose(dstrengths)[:, k]
            dA = tf.sparse_to_dense(tf.where(tf.equal(adj, 1)), adj.shape, k_dstrengths)
            A_rowsum = tf.reshape(tf.reduce_sum(A, axis=1), [-1, 1])
            dA_rowsum = tf.reshape(tf.reduce_sum(dA, axis=1), [-1, 1])
            rec = tf.pow(A_rowsum, -2)
            dQk = tf.multiply(tf.multiply((1-alpha), rec),
                              tf.subtract(tf.multiply(A_rowsum, dA),
                                          tf.multiply(dA_rowsum, A)
                                          )
                              )
            while True:
                t += 1
                dp_new = tf.add(tf.tensordot(dp[:, k], Q, 1), tf.tensordot(p, dQk, 1))
                pre = dp[:, k]
                dp = tf.concat(values=[dp[:, :k], tf.reshape(dp_new, [-1, 1]), dp[:, (k+1):]], axis=1)
                if tf.reduce_max(tf.abs(tf.subtract(pre, dp[:, k]))).eval() < epsilon or t > max_iter:
                    break
    return dp


def objective_function(adj, features, vertices, sources, destinations, alpha, max_iter,
                       lambda_param, epsilon, small_epsilon, margin_loss, w):
    """ Objective function for minimization in the training process

    :param adj: adjacency matrix
    :type adj: numpy.array
    :param features: feature vector
    :type features: numpy.array
    :param w: parameter vector
    :type w: numpy.array
    :param vertices: list of indices of all nodes
    :type vertices: list(int)
    :param sources: list of indices of source nodes
    :type sources: list(int)
    :param destinations: list of indices of destination nodes
    :type destinations: list(list(int))
    :param alpha: restart probability
    :type alpha: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :param lambda_param: regularization parameter
    :type lambda_param: float
    :param epsilon: tolerance parameter for page rank convergence
    :type epsilon: float
    :param small_epsilon: change parameter in strengths of the edges
    :rtype small_epsilon: float
    :type small_epsilon: float
    :param margin_loss: margin loss
    :type margin_loss: float
    :return: value of the objective function for the given parameters
    :rtype: float
    """
    tf.reset_default_graph()
    features = tf.constant(features, dtype=tf.float32)
    w = tf.constant(w, dtype=tf.float32)
    adj = tf.constant(adj, dtype=tf.float32)
    margin_loss = tf.constant(margin_loss, dtype=tf.float32)
    strengths = logistic_edge_strength_function(w, features) + small_epsilon
    A = tf.sparse_to_dense(tf.where(tf.greater(adj, 0.0)), adj.shape, strengths)
    Q_prim = get_stochastic_transition_matrix(A)
    loss = tf.constant(0.0)
    for source, i in zip(sources, range(len(sources))):
        print('{} Objective func, source {}, index {}'.format(time.strftime("%c"), source, i))
        Q = get_transition_matrix(Q_prim, source, alpha)
        p = iterative_page_rank(Q, epsilon, max_iter)
        l_set = list(set(vertices) - set(destinations[i] + [source]))
        diff = get_differences(p, l_set, destinations[i])
        h = loss_function(diff, margin_loss)
        loss = tf.add(loss, tf.reduce_sum(h))

    with tf.Session():
        return (tf.reduce_sum(tf.square(w)) + lambda_param * loss).eval()


def gradient_function(adj, features, vertices, sources, destinations, alpha, max_iter,
                      lambda_param, epsilon, small_epsilon, margin_loss, w):
    """ Gradient function

    :param adj: adjacency matrix
    :type adj: numpy.array
    :param features: feature vector
    :type features: numpy.array
    :param w: parameter vector
    :type w: numpy.array
    :param vertices: list of indices of all nodes
    :type vertices: list(int)
    :param sources: list of indices of source nodes
    :type sources: list(int)
    :param destinations: list of indices of destination nodes
    :type destinations: list(list(int))
    :param alpha: restart probability
    :type alpha: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :param lambda_param: regularization parameter
    :type lambda_param: float
    :param epsilon: tolerance parameter for page rank convergence
    :type epsilon: float
    :param small_epsilon: change parameter in strengths of the edges
    :rtype small_epsilon: float
    :type small_epsilon: float
    :param margin_loss: margin loss
    :type margin_loss: float
    :return: values of the gradient function
    :rtype: numpy.array
    """
    tf.reset_default_graph()
    features = tf.constant(features, dtype=tf.float32)
    w = tf.constant(w, dtype=tf.float32)
    adj = tf.constant(adj, dtype=tf.float32)
    margin_loss = tf.constant(margin_loss, dtype=tf.float32)
    lambda_param = tf.constant(lambda_param, dtype=tf.float32)
    gr = tf.zeros(w.shape[0], dtype=tf.float32)
    strengths = logistic_edge_strength_function(w, features) + small_epsilon
    A = tf.sparse_to_dense(tf.where(tf.equal(adj, 1)), adj.shape, strengths)
    Q_prim = get_stochastic_transition_matrix(A)
    for source, i in zip(sources, range(len(sources))):
        print('{} Gradient func, source {}, index {}'.format(time.strftime("%c"), source, i))
        Q = get_transition_matrix(Q_prim, source, alpha)
        p = iterative_page_rank(Q, epsilon, max_iter)
        dp = iterative_page_rank_derivative(adj, p, Q, A, epsilon, max_iter, w, features, alpha)
        l_set = list(set(vertices) - set(destinations[i] + [source]))
        diff = get_differences(p, l_set, destinations[i])
        dh = derivative_logistic_function(diff, margin_loss)
        for k in range(w.shape[0]):
            value = tf.add(tf.multiply(w[k], 2.0),
                           tf.multiply(lambda_param,
                                       tf.reduce_sum(tf.multiply(dh, get_differences(dp[:, k], l_set, destinations[i]))
                                                     )
                                       )
                           )
            gr = tf.add(gr, tf.sparse_to_dense([k], (w.shape[0], ), value))
    with tf.Session():
        return gr.eval().astype(np.float64)


def supervised_random_walks(graph, sources, destinations, alpha=0.3, lambda_par=1, margin_loss=0.4, max_iter=100):
    """ Calculates the optimized parameter vector w with supervised random walk algorithm
    (lBFGS-b) from directed network

    :param graph: igraph object
    :type graph: igraph.Graph
    :param sources: list of indices of source nodes
    :type sources: list(int)
    :param destinations: list of indices of destination nodes
    :type destinations: list(list(int))
    :param alpha: restart probability
    :type alpha: float
    :param lambda_par: regularization parameter
    :type lambda_par: float
    :param margin_loss: margin loss
    :type margin_loss: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :return: result of the optimization with the optimized parameter vector w, optimized function value,
    and information dictionary:
        - d['warnflag'] is
                - 0 if converged,
                - 1 if too many function evaluations or too many iterations,
                - 2 if stopped for another reason, given in d[‘task’]
        - d['grad'] is the gradient at the minimum (should be 0 ish)
        - d['funcalls'] is the number of function calls made.
        - d['nit'] is the number of iterations.
    :rtype: numpy.array, float, dict
    """
    epsilon = 1e-12
    small_epsilon = 1e-18

    features = np.array([graph.es[feature] for feature in graph.es.attributes()]).T

    w = np.ones((len(graph.es.attributes()), 1))

    adj = np.array(graph.get_adjacency().data)

    result = fmin_l_bfgs_b(functools.partial(objective_function, adj, features, graph.vs.indices, sources, destinations,
                                             alpha, max_iter, lambda_par, epsilon, small_epsilon, margin_loss),
                           w,
                           functools.partial(gradient_function, adj, features, graph.vs.indices, sources, destinations,
                                             alpha, max_iter, lambda_par, epsilon, small_epsilon, margin_loss),
                           callback=callback_func)

    return result


def callback_func(x):
    with open('results.txt', 'a') as file:
        file.write(time.strftime("%c") + '\t' + '\t'.join(str(item) for item in x.tolist()) + '\n')


def random_walks(graph, parameters, sources, alpha=0.3, max_iter=100):
    """ Random walk with given parameters and directed graph

    :param graph: igraph object
    :type graph: igraph.Graph
    :param parameters:
    :type parameters:
    :param sources: list of indices of source nodes
    :type sources: list(int)
    :param alpha: restart probability
    :type alpha: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :return: p vector for every source node
    :rtype: numpy.array
    """
    epsilon = 1e-12
    small_epsilon = 1e-18

    features = tf.transpose(tf.constant([graph.es[feature] for feature in graph.es.attributes()], dtype=tf.float32))
    parameters = tf.constant(parameters, dtype=tf.float32)

    strengths = logistic_edge_strength_function(parameters, features) + small_epsilon
    adj = tf.constant(graph.get_adjacency().data, dtype=tf.float32)
    A = tf.sparse_to_dense(tf.where(tf.equal(adj, 1)), adj.shape, strengths)
    Q_prim = get_stochastic_transition_matrix(A)
    result = tf.zeros((len(sources), Q_prim.shape[0]), dtype=tf.float32)
    for source, i in zip(sources, range(len(sources))):
        Q = get_transition_matrix(Q_prim, source, alpha)
        p = iterative_page_rank(Q, epsilon, max_iter)
        result[i] = p
    with tf.Session():
        return result.eval()
