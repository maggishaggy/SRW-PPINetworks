import time
import functools
import numpy as np
import mxnet as mx
from mxnet.context import gpu
from scipy.optimize import fmin_l_bfgs_b


def logistic_function(x, b):
    """
    Logistic function - Wilcoxon-Mann-Whitney (WMW) loss
    """
    return 1.0 / (1 + mx.nd.exp(-1 * x / b))


def derivative_logistic_function(x, b):
    """
    Derivative of logistic function
    """
    return (logistic_function(x, b) * (1 - logistic_function(x, b)))/b


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
    with gpu(0):
        pd = p.asnumpy()[d_set]
        pl = p.asnumpy()[l_set]
        result = mx.nd.array([l - d for d in pd for l in pl])
    return result


def logistic_edge_strength_function(w, features):
    """ Logistic edge strength function

    :param w: weight vector
    :type w: numpy.array
    :param features: feature vector
    :type features: numpy.array
    :return: edge strengths
    :rtype: numpy.array
    """
    return logistic_function(mx.nd.dot(features, w), 1)


def logistic_edge_strength_derivative_function(w, features):
    """ Derivative of logistic edge strength function

    :param w: weight vector
    :type w: numpy.array
    :param features: feature vector
    :type features: numpy.array
    :return: edge strengths
    :rtype: numpy.array
    """
    return mx.nd.multiply(derivative_logistic_function(mx.nd.transpose(mx.nd.dot(features, w)), 1),
                          mx.nd.transpose(features))


def get_stochastic_transition_matrix(A):
    """ Calculates the stochastic transition matrix

    :param A: adjacency matrix with edge strengths
    :type A: numpy.array
    :return: stochastic transition matrix
    :rtype: numpy.array
    """
    with gpu(0):
        Q_prim = 1 / mx.nd.sum(A, axis=1).reshape((-1, 1)) * A
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
    with gpu(0):
        one = mx.nd.zeros(Q_prim.shape)
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
    with gpu(0):
        p = mx.nd.ones((1, trans.shape[0])) / trans.shape[0]
        p_new = mx.nd.dot(p, trans)
        while not (mx.nd.max(mx.nd.abs(p - p_new)).asnumpy()[0] < epsilon or max_iter <= 0):
            p = p_new
            p_new = mx.nd.dot(p, trans)
            max_iter -= 1
    return p_new[0]


def iterative_page_rank_derivative(graph, p, Q, A, epsilon, max_iter, w, features, alpha):
    """ Derivative of PageRank vector p

    :param graph: igraph object
    :type graph: igraph.Graph
    :param p: PageRank vector
    :param p: numpy.array
    :param Q: transition matrix
    :type Q: numpy.array
    :param A: adjacency strength matrix
    :type A: numpy.array
    :param epsilon: tolerance parameter
    :type epsilon: float
    :param max_iter: maximum number of iterations
    :type max_iter: int
    :param w: parameter vector
    :type w: numpy.array
    :param features: feature vector
    :type features: numpy.array
    :param alpha: restart probability
    :type alpha: float
    :return: derivative of PageRank vector
    :rtype: numpy.array
    """
    with gpu(0):
        dp = mx.nd.zeros((Q.shape[0], w.shape[0]))
        dstrengths = logistic_edge_strength_derivative_function(w, features)
        for k in range(w.shape[0]):
            t = 0
            graph.es['temp'] = mx.nd.transpose(dstrengths)[:, k].reshape((-1,)).asnumpy().tolist()
            dA = mx.nd.array(graph.get_adjacency(attribute='temp').data)
            A_rowsum = mx.nd.sum(A, axis=1).reshape((-1, 1))
            dA_rowsum = mx.nd.sum(dA, axis=1).reshape((-1, 1))
            rec = mx.nd.power(mx.nd.sum(A, axis=1).reshape((-1, 1)), -2)
            dQk = (1 - alpha) * rec * ((A_rowsum * dA) - (dA_rowsum * A))
            while True:
                t += 1
                dp_new = mx.nd.dot(dp[:, k], Q) + mx.nd.dot(p, dQk)
                pre = mx.nd.array(dp[:, k])
                dp[:, k] = dp_new
                if mx.nd.max(mx.nd.abs(pre - dp[:, k])).asnumpy()[0] < epsilon or t > max_iter:
                    break
    return dp


def objective_function(graph, features, sources, destinations, alpha, max_iter,
                       lambda_param, epsilon, small_epsilon, margin_loss, w):
    """ Objective function for minimization in the training process

    :param graph: igraph object
    :type graph: igraph.Graph
    :param features: feature vector
    :type features: numpy.array
    :param w: parameter vector
    :type w: numpy.array
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
    with gpu(0):
        features = mx.nd.array(features)
        w = mx.nd.array(w)
        strengths = logistic_edge_strength_function(w, features) + small_epsilon
        graph.es['strength'] = strengths.reshape((-1,)).asnumpy()
        A = mx.nd.array(graph.get_adjacency(attribute='strength').data)
        Q_prim = get_stochastic_transition_matrix(A)
        loss = 0
        for source, i in zip(sources, range(len(sources))):
            print('{} Objective func, source {}, index {}'.format(time.strftime("%c"), source, i))
            Q = get_transition_matrix(Q_prim, source, alpha)
            p = iterative_page_rank(Q, epsilon, max_iter)
            l_set = list(set(graph.vs.indices) - set(destinations[i] + [source]))
            diff = get_differences(p, l_set, destinations[i])
            h = loss_function(diff, margin_loss)
            loss += mx.nd.sum(h).asnumpy()[0]
    return float(mx.nd.sum(mx.nd.square(w)).asnumpy()[0] + lambda_param * loss)


def gradient_function(graph, features, sources, destinations, alpha, max_iter,
                      lambda_param, epsilon, small_epsilon, margin_loss, w):
    """ Gradient function

    :param graph: igraph object
    :type graph: igraph.Graph
    :param features: feature vector
    :type features: numpy.array
    :param w: parameter vector
    :type w: numpy.array
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
    with gpu(0):
        features = mx.nd.array(features)
        w = mx.nd.array(w)
        gr = mx.nd.zeros(w.shape[0])
        strengths = logistic_edge_strength_function(w, features) + small_epsilon
        graph.es['strength'] = strengths.reshape((-1,)).asnumpy()
        A = mx.nd.array(graph.get_adjacency(attribute='strength').data)
        Q_prim = get_stochastic_transition_matrix(A)
        for source, i in zip(sources, range(len(sources))):
            print('{} Gradient func, source {}, index {}'.format(time.strftime("%c"), source, i))
            Q = get_transition_matrix(Q_prim, source, alpha)
            p = iterative_page_rank(Q, epsilon, max_iter)
            dp = iterative_page_rank_derivative(graph, p, Q, A, epsilon, max_iter, w, features, alpha)
            l_set = list(set(graph.vs.indices) - set(destinations[i] + [source]))
            diff = get_differences(p, l_set, destinations[i])
            dh = derivative_logistic_function(diff, margin_loss)
            for k in range(w.shape[0]):
                gr[k] += 2 * w[k] + lambda_param * mx.nd.sum(dh * get_differences(dp[:, k], l_set, destinations[i]))
    return gr.asnumpy().astype(np.float64)


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

    result = fmin_l_bfgs_b(functools.partial(objective_function, graph, features, sources, destinations, alpha,
                                             max_iter, lambda_par, epsilon, small_epsilon, margin_loss),
                           w,
                           functools.partial(gradient_function, graph, features, sources, destinations, alpha,
                                             max_iter, lambda_par, epsilon, small_epsilon, margin_loss),
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
    with gpu(0):
        epsilon = 1e-12
        small_epsilon = 1e-18

        features = mx.nd.array([graph.es[feature] for feature in graph.es.attributes()]).T
        parameters = mx.nd.array(parameters)

        strengths = logistic_edge_strength_function(parameters, features) + small_epsilon
        graph.es['strength'] = strengths.reshape((-1,)).asnumpy()
        A = mx.nd.array(graph.get_adjacency(attribute='strength').data)
        Q_prim = get_stochastic_transition_matrix(A)
        result = mx.nd.zeros((len(sources), Q_prim.shape[0]))
        for source, i in zip(sources, range(len(sources))):
            Q = get_transition_matrix(Q_prim, source, alpha)
            p = iterative_page_rank(Q, epsilon, max_iter)
            result[i] = p

    return result.asnumpy()
