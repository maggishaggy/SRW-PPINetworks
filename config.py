class Config:
    def __init__(self, num_vertices, num_features, alpha, max_iter, lambda_param, epsilon,
                 small_epsilon, margin_loss, summary_dir, save_dir, num_epochs=100,
                 save_period=1000, initial_learning_rate=0.01, learning_rate_decay_factor=1.0,
                 num_steps_per_decay=100000, clip_gradients=5.0, num_classes=0):
        """

        :param num_features: number of features in the feature vector
        :type num_features: int
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
        :param summary_dir: directory to save the summary info
        :type summary_dir: str
        :param num_epochs: number of epochs
        :type num_epochs: int
        :param initial_learning_rate: the initial value for the learning rate
        :type initial_learning_rate: float
        :param learning_rate_decay_factor: learning rate decay factor
        :type learning_rate_decay_factor: float
        :param num_steps_per_decay: number of steps per decay
        :type num_steps_per_decay: int
        :param clip_gradients: parameter for clipping gradients
        :type clip_gradients: float
        :param num_classes: number of GO terms as classes
        :type num_classes: int
        """
        self.num_vertices = num_vertices
        self.num_features = num_features
        self.alpha = alpha
        self.max_iter = max_iter
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.small_epsilon = small_epsilon
        self.margin_loss = margin_loss
        self.summary_dir = summary_dir
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        self.save_period = save_period
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.num_steps_per_decay = num_steps_per_decay
        self.clip_gradients = clip_gradients
        self.num_classes = num_classes
