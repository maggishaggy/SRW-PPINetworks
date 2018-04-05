import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def logistic_function(x, b):
    """
    Logistic function - Wilcoxon-Mann-Whitney (WMW) loss
    """
    return 1.0 / (1 + np.exp(-1 * x / b))


def get_differences(p, l_set, d_set):
    """ Calculate difference between pl and pd

    :param p: stationary distribution
    :type p: numpy.array
    :param l_set: indices of the nodes in L set
    :type l_set: list
    :param d_set: indices of the nodes in D set
    :type d_set: list
    :return: difference between pl and pd
    :rtype: numpy.array
    """
    pd = p[d_set]
    pl = p[l_set]
    return np.array([l - d for d in pd for l in pl])


def logistic_edge_strength_function(w, features):
    """ Logistic edge strength function

    :param w: weight vector
    :type w: numpy.array
    :param features: feature vector
    :type features: numpy.array
    :return: edge strengths
    :rtype: numpy.array
    """
    return logistic_function(features.dot(w), 1)


def get_stochastic_transition_matrix(A):
    """ Calculates the stochastic transition matrix

    :param A: adjacency matrix with edge strengths
    :type A: numpy.array
    :return: stochastic transition matrix
    :rtype: numpy.array
    """
    Q_prim = 1 / np.sum(A, axis=1).reshape(-1, 1) * A
    return Q_prim


def get_transition_matrix(Q_prim, start_node, alpha):
    """ Calculate the transition matrix from given stochastic transition matrix,
    start node and restart probability

    :param Q_prim: stochastic transition matrix
    :type Q_prim: numpy.array
    :param start_node: index of the start node
    :type start_node: int
    :param alpha: restart probability
    :type alpha: float
    :return: transition matrix
    :rtype: numpy.array
    """
    one = np.zeros(Q_prim.shape)
    one[:, start_node] = 1
    return (1 - alpha) * Q_prim + alpha * one


def loss_function(diff, b):
    """ Loss function

    :param diff: differences pl - pd
    :type diff: numpy.array
    :param b: parameter in logistic function
    :type b: float
    :return: loss for each difference
    :rtype: numpy.array
    """
    return logistic_function(diff, b)


def iterative_page_rank(trans, epsilon, max_iter):
    """ Iterative power-iterator like computation of PageRank vector p

    :param trans: transition matrix
    :type trans: numpy.array
    :param epsilon: tolerance parameter
    :type epsilon: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :return: stationary distribution
    :rtype: numpy.array
    """
    p = np.sum(trans > 0, axis=1)
    p_new = np.dot(p, trans)
    while not (np.allclose(p, p_new, rtol=0, atol=epsilon) or max_iter <= 0):
        p = p_new
        p_new = np.dot(p, trans)
        max_iter -= 1
    return p_new


def objective_function(graph, features, w, sources, destinations, alpha, max_iter,
                       lambda_param, epsilon, small_epsilon, margin_loss):
    """ Objective function for minimization in the training process

    :param graph: igraph object
    :type graph: igraph.Graph
    :param features:
    :param w: parameter vector
    :type w: numpy.array
    :param sources: list of indices of source nodes
    :type sources: list(int)
    :param destinations: list of indices of destination nodes
    :type destinations: list(int)
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
    strengths = logistic_edge_strength_function(w, features) + small_epsilon
    graph.es['strength'] = strengths.flatten()
    A = np.array(graph.get_adjacency(attribute='strength').data)
    Q_prim = get_stochastic_transition_matrix(A)
    loss = 0
    for source, i in zip(sources, range(len(sources))):
        Q = get_transition_matrix(Q_prim, source, alpha)
        p = iterative_page_rank(Q, epsilon, max_iter)
        l_set = list(set(graph.vs.indices) - set(destinations[i] + [source]))
        diff = get_differences(p, l_set, destinations)
        h = loss_function(diff, margin_loss)
        loss += np.sum(h)
    return np.sum(np.square(w)) + lambda_param * loss


def supervised_random_walks(graph, sources, destinations, alpha=0.3, lambda_par=1, margin_loss=0.4, max_iter=100):
    """ Calculates the optimized parameter vector w with supervised random walk algorithm

    :param graph: igraph object
    :type graph: igraph.Graph
    :param sources: list of indices of source nodes
    :type sources: list(int)
    :param destinations: list of indices of destination nodes
    :type destinations: list(int)
    :param alpha: restart probability
    :type alpha: float
    :param lambda_par: regularization parameter
    :type lambda_par: float
    :param margin_loss: margin loss
    :type margin_loss: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :return: optimized parameter vector w
    :rtype: numpy.array
    """
    epsilon = 1e-12
    small_epsilon = 1e-18

    features = np.array([graph.es[feature] for feature in graph.es.attributes()]).T

    w = np.ones((len(graph.es.attributes()), 1))

    objective_function(graph, features, w, sources, destinations, alpha,
                       max_iter, lambda_par, epsilon, small_epsilon, margin_loss)

    w_optimized = ...

    return w_optimized
