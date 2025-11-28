import numpy as np
import tensorflow as tf
from Algorithm.utils import normalization, renormalization, rounding
from Algorithm.utils import xavier_init
from Algorithm.utils import binary_sampler, uniform_sampler, sample_batch_index
from progress_bar import InitBar
from tqdm import tqdm

class GAIN:
    def __init__(self, gain_parameters, Initial_num, thre_value, epsilon, value, s_miss, learning_rate):
        self.gain_parameters = gain_parameters
        self.Initial_num = Initial_num
        self.thre_value = thre_value
        self.epsilon = epsilon
        self.value = value
        self.s_miss = s_miss
        self.learning_rate = learning_rate

        self.CLIP = [-0.01, 0.01]

        # System parameters
        self.batch_size = gain_parameters['batch_size']
        self.hint_rate = gain_parameters['hint_rate']
        self.alpha = gain_parameters['alpha']
        self.epoch = gain_parameters['epoch']
        self.guarantee = gain_parameters['guarantee']
        self.Sinkhorn_iter = 20

        self.h_dim = None
        self.no = None
        self.dim = None

        self.is_trained_ = False

        # Initialize TensorFlow graph
        self._build_model()

    def _build_model(self):
        tf.compat.v1.disable_eager_execution()
        self.X = tf.compat.v1.placeholder(tf.float32, shape=[None, None])  # Placeholder for data (dimension unknown initially)
        self.M = tf.compat.v1.placeholder(tf.float32, shape=[None, None])  # Placeholder for mask
        self.H = tf.compat.v1.placeholder(tf.float32, shape=[None, None])  # Placeholder for hint
        self.n_num = tf.Variable(1)

        # Discriminator variables
        self.D_W1 = tf.Variable(xavier_init([None, self.h_dim]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        self.D_W2 = tf.Variable(xavier_init([self.h_dim, self.h_dim]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        self.D_W3 = tf.Variable(xavier_init([self.h_dim, None]))  # Outputs size matches data dimension
        self.D_b3 = tf.Variable(tf.zeros(shape=[None]))

        self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]

        # Generator variables
        self.G_W1 = tf.Variable(xavier_init([None, self.h_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        self.G_W2 = tf.Variable(xavier_init([self.h_dim, self.h_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        self.G_W3 = tf.Variable(xavier_init([self.h_dim, None]))  # Outputs size matches data dimension
        self.G_b3 = tf.Variable(tf.zeros(shape=[None]))

        self.theta_G = [self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3]

        # Build cost functions and Sinkhorn loss
        self._define_cost_and_loss()

    def _define_cost_and_loss(self):
        def cost_matrix(x, y, m, p=2):
            "Returns the cost matrix C_{ij}=|x_i - y_j|^p"
            x_col = tf.expand_dims(x, 1)
            m_x_col = tf.expand_dims(m, 1)
            y_lin = tf.expand_dims(y, 0)
            m_y_lin = tf.expand_dims(m, 0)
            c = tf.reduce_sum((tf.abs(x_col * m_x_col - y_lin * m_y_lin)) ** p, axis=2)
            return c

        def sinkhorn_loss(x, y, m, epsilon, n, niter, p=2):
            C = cost_matrix(x, y, m, p=p)
            mu = tf.constant(1.0 / n, shape=[n])
            nu = tf.constant(1.0 / n, shape=[n])

            def M(u, v):
                return (-C + tf.expand_dims(u, 1) + tf.expand_dims(v, 0)) / epsilon

            def lse(A):
                return tf.reduce_logsumexp(A, axis=1, keepdims=True)

            u, v = 0. * mu, 0. * nu
            for i in range(niter):
                u = epsilon * (tf.math.log(mu) - tf.squeeze(lse(M(u, v)))) + u
                v = epsilon * (tf.math.log(nu) - tf.squeeze(lse(tf.transpose(M(u, v))))) + v

            u_final, v_final = u, v
            pi = tf.exp(M(u_final, v_final))
            cost = tf.reduce_sum(pi * C)
            return cost


        # Define the generator and discriminator
        self.G_sample = self.generator(self.X, self.M)
        self.Sinkhorn_loss = sinkhorn_loss(self.X, self.G_sample, self.M, self.epsilon, self.batch_size, self.Sinkhorn_iter)
        self.Hat_X = self.X * self.M + self.G_sample * (1 - self.M)
        self.D_prob = self.discriminator(self.Hat_X, self.H)

        self.D_loss_temp = -tf.reduce_mean(self.M * (self.D_prob + 1e-8) + (1 - self.M) * (1. - self.D_prob + 1e-8))
        self.G_loss_temp = -tf.reduce_mean((1 - self.M) * (self.D_prob + 1e-8))
        self.MSE_loss = tf.reduce_mean((self.M * self.X - self.M * self.G_sample) ** 2) / tf.reduce_mean(self.M)
        self.D_loss = self.D_loss_temp
        self.G_loss = self.G_loss_temp + self.value * self.Sinkhorn_loss

    def generator(self, x, m):
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, self.G_W3) + self.G_b3)
        return G_prob

    def discriminator(self, x, h):
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
        D_logit = tf.matmul(D_h2, self.D_W3) + self.D_b3
        return D_logit

    def train(self, data, missing_mask):
        # Normalize input data
        norm_data, norm_parameters = normalization(data)
        norm_data_x = np.nan_to_num(norm_data, 0)

        data_m = 1 - np.isnan(data)

        self.no, self.dim = norm_data_x.shape
        self.h_dim = int(self.dim)

        D_solver = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.D_loss, var_list=self.theta_D)
        G_solver = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.G_loss, var_list=self.theta_G)

        clip_d_op = [var.assign(tf.clip_by_value(var, self.CLIP[0], self.CLIP[1])) for var in self.theta_D]

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        # Training loop
        pbar = InitBar()
        num = 0
        for it in tqdm(range(self.gain_parameters['epoch'])):
            data_list = sample_batch_index(len(norm_data_x), len(norm_data_x))
            for mb_idx in data_list:
                num += 1
                pbar(num / (self.gain_parameters['epoch'] * len(data_list)) * 100)
                X_mb = norm_data_x[mb_idx, :]
                M_mb = missing_mask[mb_idx, :]
                Z_mb = uniform_sampler(0, 0.01, self.batch_size, self.dim)
                H_mb_temp = binary_sampler(self.hint_rate, self.batch_size, self.dim)
                H_mb = M_mb * H_mb_temp
                X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
                _, D_loss_curr = sess.run([D_solver, self.D_loss_temp], feed_dict={self.M: M_mb, self.X: X_mb, self.H: H_mb})
                sess.run(clip_d_op)
                _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, self.G_loss_temp, self.MSE_loss],
                                                         feed_dict={self.X: X_mb, self.M: M_mb, self.H: H_mb})
        self.is_trained_ = True
        return self

    def predict(self, data, missing_mask):
        if not self.is_trained_:
            raise RuntimeError("This MISS instance is not trained yet. Call 'train' with appropriate data before using 'predict'.")
        norm_data, norm_parameters = normalization(data)
        norm_data_x = np.nan_to_num(norm_data, 0)

        Z_mb = uniform_sampler(0, 0.01, len(data), self.dim)
        M_mb = missing_mask
        X_mb = norm_data_x
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        imputed_data = sess.run([self.G_sample], feed_dict={self.X: X_mb, self.M: M_mb})[0]

        imputed_data = missing_mask * norm_data_x + (1 - missing_mask) * imputed_data
        imputed_data = renormalization(imputed_data, norm_parameters)
        imputed_data = rounding(imputed_data, data)

        return imputed_data
