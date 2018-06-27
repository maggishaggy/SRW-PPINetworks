import os
import copy
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf


class SRW:
    def __init__(self, config, mode):
        """ Initialization of the model

        :param config: configuration parameters
        :type config: Config object
        """
        assert mode in ['training', 'inference']
        self.config = config
        self.mode = mode
        self.build()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.mode == 'training':
            self.build_optimizer()
            self.build_summary()

    @staticmethod
    def logistic_function(x, b):
        """
        Logistic function - Wilcoxon-Mann-Whitney (WMW) loss
        """
        return 1.0 / (1 + tf.exp(-1 * x / b))

    @staticmethod
    def get_differences(p, l_set, d_set):
        """ Calculate difference between pl and pd

        :param p: stationary distribution
        :type p: tensorflow.Tensor
        :param l_set: indices of the nodes in L set
        :type l_set: tensorflow.Tensor
        :param d_set: indices of the nodes in D set
        :type d_set: tensorflow.Tensor
        :return: difference between pl and pd
        :rtype: tensorflow.Tensor
        """
        pd = tf.gather(p, d_set)
        pl = tf.gather(p, l_set)
        result = tf.reshape(tf.subtract(tf.expand_dims(pl, 0), tf.expand_dims(pd, 1)), [-1])
        return result

    def logistic_edge_strength_function(self, w, features):
        """ Logistic edge strength function

        :param w: weight vector
        :type w: tensorflow.Variable
        :param features: feature vector
        :type features: tensorflow.Tensor
        :return: edge strengths
        :rtype: tensorflow.Tensor
        """
        return self.logistic_function(tf.tensordot(features, w, 1), 1)

    @staticmethod
    def get_stochastic_transition_matrix(A):
        """ Calculates the stochastic transition matrix

        :param A: adjacency matrix with edge strengths
        :type A: tensorflow.Tensor
        :return: stochastic transition matrix
        :rtype: tensorflow.Tensor
        """
        Q_prim = tf.multiply(1 / tf.reduce_sum(A, axis=1, keep_dims=True), A)
        return Q_prim

    @staticmethod
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
        # one = np.zeros(Q_prim.shape, dtype=np.float32)
        # one[:, start_node] = 1.0
        # one = tf.constant(one)
        one = tf.concat([tf.zeros([Q_prim.shape[0], start_node[0]-1], dtype=tf.float32),
                         tf.ones([Q_prim.shape[0], 1], dtype=tf.float32),
                         tf.zeros([Q_prim.shape[0], -start_node[0] + Q_prim.shape[1]], dtype=tf.float32)],
                        axis=1)
        return tf.add((1 - alpha) * Q_prim, alpha * one)

    def loss_function(self, diff, b):
        """ Loss function

        :param diff: differences pl - pd
        :type diff: tensorflow.Tensor
        :param b: parameter in logistic function
        :type b: float
        :return: loss for each difference
        :rtype: tensorflow.Tensor
        """
        return self.logistic_function(diff, b)

    @staticmethod
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
        def f1(): return 1

        def f2(): return 0

        p = tf.divide(tf.ones((1, trans.shape[0])), tf.cast(trans.shape[0], tf.float32))
        p_new = tf.tensordot(p, trans, 1)
        for i in range(max_iter):
            p = p_new
            p_new = tf.tensordot(p, trans, 1)
            if tf.cond(tf.reduce_max(tf.abs(tf.subtract(p, p_new))) < epsilon, f1, f2) == 1:
                break
        return p_new[0]

    def build(self):
        features = tf.placeholder(dtype=tf.float32, shape=[None,
                                                           self.config.num_features])
        w = tf.Variable(tf.ones((self.config.num_features, 1)))
        adj = tf.placeholder(dtype=tf.float32, shape=[self.config.num_vertices,
                                                      self.config.num_vertices])
        source = tf.placeholder(dtype=tf.int32, shape=[1])

        strengths = self.logistic_edge_strength_function(w, features) + self.config.small_epsilon
        A = tf.SparseTensor(tf.where(tf.greater(adj, 0.0)), strengths[:, 0], dense_shape=adj.shape)
        # hack for bug in tensorflow because sparse_tensor_to_dense() does not have gradient
        A = tf.sparse_add(tf.zeros(A.dense_shape), A)
        Q_prim = self.get_stochastic_transition_matrix(A)
        Q = self.get_transition_matrix(Q_prim, source, self.config.alpha)
        p = self.iterative_page_rank(Q, self.config.epsilon, self.config.max_iter)

        if self.mode == 'training':
            vertices = tf.placeholder(dtype=tf.int32, shape=[None])
            destinations = tf.placeholder(dtype=tf.int32, shape=[None])
            l_set = tf.sets.set_difference(tf.expand_dims(vertices, axis=0),
                                           tf.sets.set_union(tf.expand_dims(destinations, axis=0),
                                                             tf.expand_dims(source, axis=0)))
            l_set = tf.sparse_tensor_to_dense(l_set)[0]
            diff = self.get_differences(p, l_set, destinations)
            loss = self.loss_function(diff, self.config.margin_loss)
            self.loss = tf.reduce_sum(loss)
            self.vertices = vertices
            self.destinations = destinations
        else:
            self.result = p

        self.features = features
        self.adj = adj
        self.source = source

    def train(self, sess, train_data):
        print("Training the model...")

        if not os.path.exists(self.config.summary_dir):
            os.mkdir(self.config.summary_dir)
        train_writer = tf.summary.FileWriter(self.config.summary_dir,
                                             sess.graph)

        for _ in tqdm(list(range(self.config.num_epochs)), desc='epoch'):
            features = train_data['features']
            adj = train_data['adj']
            vertices = train_data['vertices']
            for i, _ in enumerate(tqdm(list(range(train_data['num_sources'])), desc='source')):
                source, destinations = train_data['data'][i]['source'], train_data['data'][i]['destinations']
                feed_dict = {self.features: features,
                             self.adj: adj,
                             self.vertices: vertices,
                             self.source: [source],
                             self.destinations: destinations}
                _, summary, global_step = sess.run([self.opt_op,
                                                    self.summary,
                                                    self.global_step],
                                                   feed_dict=feed_dict)
                if (global_step + 1) % self.config.save_period == 0:
                    self.save()
                train_writer.add_summary(summary, global_step)

        self.save()
        train_writer.close()
        print("Training complete.")

    def predict(self, sess, inference_data):
        print("Predicting ...")
        features = inference_data['features']
        adj = inference_data['adj']
        predictions = dict()
        for i, _ in enumerate(tqdm(list(range(inference_data['num_sources'])), desc='source')):
            source = inference_data['data'][i]['source']
            feed_dict = {self.features: features,
                         self.adj: adj,
                         self.source: [source]}
            result = sess.run([self.result], feed_dict=feed_dict)
            predictions[source] = result

        print("Prediction compete.")
        return predictions

    def build_optimizer(self):
        learning_rate = tf.constant(self.config.initial_learning_rate)
        if self.config.learning_rate_decay_factor < 1.0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=self.config.num_steps_per_decay,
                    decay_rate=self.config.learning_rate_decay_factor,
                    staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = None
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.initial_learning_rate)
            opt_op = tf.contrib.layers.optimize_loss(
                loss=self.loss,
                global_step=self.global_step,
                learning_rate=learning_rate,
                optimizer=optimizer,
                clip_gradients=self.config.clip_gradients,
                learning_rate_decay_fn=learning_rate_decay_fn)

        self.opt_op = opt_op

    @staticmethod
    def variable_summary(var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("loss", self.loss)

        self.summary = tf.summary.merge_all()

    def save(self):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path + ".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            info_file = open(info_path, "rb")
            config = pickle.load(info_file)
            global_step = config.global_step
            info_file.close()
            save_path = os.path.join(config.save_dir,
                                     str(global_step) + ".npy")

        print("Loading the model from %s..." % save_path)
        data_dict = np.load(save_path).item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." % count)
