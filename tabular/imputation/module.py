import numpy as np
import tensorflow as tf
import os


# =============================================================================
# 1. 基础数据处理工具 (Shared)
# =============================================================================

def normalization(data, parameters=None):
    """归一化数据到 [0, 1]"""
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)
        norm_parameters = {'min_val': min_val, 'max_val': max_val}
    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    """反归一化"""
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']
    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]
    return renorm_data


def rounding(imputed_data, data_x):
    """对分类变量取整"""
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()
    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
    return rounded_data


def sample_batch_index(total, batch_size):
    """生成随机 Batch 索引"""
    total_idx = np.random.permutation(total)
    return total_idx[:batch_size]


# =============================================================================
# 2. 神经网络初始化与采样工具 (Shared)
# =============================================================================

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random.normal(shape=size, stddev=xavier_stddev)


def binary_sampler(p, rows, cols):
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    return 1 * (unif_random_matrix < p)


def uniform_sampler(low, high, rows, cols):
    return np.random.uniform(low, high, size=[rows, cols])


# =============================================================================
# 3. 核心计算组件 (Network & Loss)
# =============================================================================

def sinkhorn_loss(x, y, m, epsilon, n, niter=20, p=2):
    """计算 Sinkhorn 距离损失"""
    # Cost matrix
    x_col = tf.expand_dims(x, 1)
    m_x_col = tf.expand_dims(m, 1)
    y_lin = tf.expand_dims(y, 0)
    m_y_lin = tf.expand_dims(m, 0)
    C = tf.reduce_sum((tf.abs(x_col * m_x_col - y_lin * m_y_lin)) ** p, axis=2)

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

    pi = tf.exp(M(u, v))
    cost = tf.reduce_sum(pi * C)
    return cost


def create_network_variables(dim, h_dim):
    """创建 Generator 和 Discriminator 的权重变量"""
    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator variables
    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    return theta_D, theta_G


def generator_net(x, m, params):
    """标准生成器网络"""
    [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3] = params
    inputs = tf.concat(values=[x, m], axis=1)
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
    return G_prob


def generator_net_custom(x, m, params_list):
    """SCIS 专用的影子生成器 (支持直接传入权重列表)"""
    inputs = tf.concat(values=[x, m], axis=1)
    # 这里的 params_list 顺序对应 weights_list
    G_h1 = tf.nn.relu(tf.matmul(inputs, params_list[0]) + params_list[3])
    G_h2 = tf.nn.relu(tf.matmul(G_h1, params_list[1]) + params_list[4])
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, params_list[2]) + params_list[5])
    return G_prob


def discriminator_net(x, h, params):
    """判别器网络"""
    [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3] = params
    inputs = tf.concat(values=[x, h], axis=1)
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    return D_logit

def xavier_init_AE(fan_in, fan_out, constant=1):
    """AutoEncoder 专用的 Xavier 初始化"""
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

def vae_initialize_weights(n_input, n_hidden_recog_1, n_hidden_recog_2,
                           n_hidden_gener_1, n_hidden_gener_2, n_z):
    """初始化 VAE 部分的权重"""
    all_weights = dict()
    # Encoder Weights
    all_weights['weights_recog'] = {
        'h1': tf.Variable(xavier_init_AE(n_input, n_hidden_recog_1)),
        'h2': tf.Variable(xavier_init_AE(n_hidden_recog_1, n_hidden_recog_2)),
        'out_mean': tf.Variable(xavier_init_AE(n_hidden_recog_2, n_z)),
        'out_log_sigma': tf.Variable(xavier_init_AE(n_hidden_recog_2, n_z))}
    all_weights['biases_recog'] = {
        'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
        'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
        'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
        'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
    # Decoder Weights
    all_weights['weights_gener'] = {
        'h1': tf.Variable(xavier_init_AE(n_z, n_hidden_gener_1)),
        'h2': tf.Variable(xavier_init_AE(n_hidden_gener_1, n_hidden_gener_2)),
        'out_mean': tf.Variable(xavier_init_AE(n_hidden_gener_2, n_input)),
        'out_log_sigma': tf.Variable(xavier_init_AE(n_hidden_gener_2, n_input))}
    all_weights['biases_gener'] = {
        'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
        'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
        'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
        'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
    return all_weights

def vae_initialize_discriminator(dim, h_dim1, h_dim2):
    """初始化 VAE-GAIN 判别器的权重"""
    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim1]))  # Data + Hint
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim1]))
    D_W2 = tf.Variable(xavier_init([h_dim1, h_dim2]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim2]))
    D_W3 = tf.Variable(xavier_init([h_dim2, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))
    return [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

def vae_recognition_network(x_in, weights, biases):
    """Encoder"""
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x_in, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
    z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])
    return z_mean, z_log_sigma_sq

def vae_generator_network(z, weights, biases):
    """Decoder (Generator)"""
    layer_1 = tf.nn.relu(tf.add(tf.matmul(z, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    x_hat_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
    x_hat_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])
    return x_hat_mean, x_hat_log_sigma_sq

def vae_discriminator_network(x, h, theta_D):
    """Discriminator"""
    [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3] = theta_D
    inputs = tf.concat(axis=1, values=[x, h])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob

# =============================================================================
# 4. 图构建器 (Build Graphs)
# =============================================================================

def build_gain_graph(dim, batch_size, epsilon, value, gpu_device):
    """
    构建 GAIN 计算图
    """
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

    X = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
    M = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
    H = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])

    theta_D, theta_G = create_network_variables(dim, int(dim))

    G_sample = generator_net(X, M, theta_G)
    Hat_X = X * M + G_sample * (1 - M)
    D_prob = discriminator_net(Hat_X, H, theta_D)

    # Losses
    sink_loss = sinkhorn_loss(X, G_sample, M, epsilon, batch_size)
    D_loss_temp = -tf.reduce_mean(M * (D_prob + 1e-8) + (1 - M) * (1. - D_prob + 1e-8))
    G_loss_temp = -tf.reduce_mean((1 - M) * (D_prob + 1e-8))

    D_loss = D_loss_temp
    G_loss = G_loss_temp + value * sink_loss
    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    # Solvers
    D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    clip_d_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in theta_D]

    return {
        'ph': {'X': X, 'M': M, 'H': H},
        'ops': {'G_sample': G_sample, 'D_solver': D_solver, 'G_solver': G_solver, 'clip_d_op': clip_d_op},
        'losses': {'MSE': MSE_loss, 'D': D_loss_temp, 'G': G_loss_temp}
    }


def build_scis_graph(dim, total_data_len, initial_num, batch_size, epsilon, value, thre_value, gpu_device):
    """
    构建 SCIS 计算图 (包含 Hessian 计算和影子生成器)
    """
    # 1. 基础部分同 GAIN
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

    X = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
    M = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
    H = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
    n_num = tf.Variable(1)  # SCIS 特有：样本量变量

    theta_D, theta_G = create_network_variables(dim, int(dim))

    G_sample = generator_net(X, M, theta_G)
    Hat_X = X * M + G_sample * (1 - M)
    D_prob = discriminator_net(Hat_X, H, theta_D)

    sink_loss = sinkhorn_loss(X, G_sample, M, epsilon, batch_size)
    D_loss_temp = -tf.reduce_mean(M * (D_prob + 1e-8) + (1 - M) * (1. - D_prob + 1e-8))
    G_loss_temp = -tf.reduce_mean((1 - M) * (D_prob + 1e-8))

    D_loss = D_loss_temp
    G_loss = G_loss_temp + value * sink_loss
    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    clip_d_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in theta_D]

    # 2. SCIS 特有部分：Hessian 与 参数扰动
    H_gradient_ = tf.gradients(G_loss, theta_G)
    Len_matrix = len(theta_G)

    # 构建快速 Hessian
    final = [tf.reshape(H_gradient_[i], [-1, 1]) for i in range(Len_matrix)]
    final_zhuan = [tf.transpose(tf.reshape(H_gradient_[i], [-1, 1])) for i in range(Len_matrix)]
    Fast_Hessians = [final[i] * final_zhuan[i] for i in range(Len_matrix)]

    # 加上单位矩阵保证可逆
    H_hessians = []
    # 注意：这里的维度计算依赖于 theta_G 的结构 (W1, W2, W3, b1, b2, b3)
    h_dim = int(dim)
    shapes = [[2 * dim * h_dim, 2 * dim * h_dim], [h_dim * h_dim, h_dim * h_dim], [h_dim * dim, h_dim * dim],
              [h_dim, h_dim], [h_dim, h_dim], [dim, dim]]

    for i in range(Len_matrix):
        flat_hess = tf.reshape(Fast_Hessians[i], shapes[i]) + tf.eye(shapes[i][0])
        H_hessians.append(flat_hess)

    H_invert = [tf.linalg.inv(item) for item in H_hessians]
    H_gradient_diag = [tf.linalg.tensor_diag_part(item) for item in H_invert]
    Mean_N = [tf.reshape(item, [-1]) for item in theta_G]

    # 方差计算
    Variance_N = [item * (1 / tf.cast(n_num, tf.float32) - 1 / total_data_len) for item in H_gradient_diag]
    Variance_n = [item * (1 / initial_num - 1 / tf.cast(n_num, tf.float32)) for item in H_gradient_diag]

    epsilon_value = (np.exp(5 / epsilon) / np.power(epsilon, np.floor(dim / 2))) ** 2

    # 生成扰动参数
    n_params_raw = [tf.random.normal(tf.shape(Mean_N[i]), Mean_N[i], epsilon_value * Variance_n[i]) for i in
                    range(Len_matrix)]
    N_params_raw = [tf.random.normal(tf.shape(n_params_raw[i]), n_params_raw[i], epsilon_value * Variance_N[i]) for i in
                    range(Len_matrix)]

    # 重塑参数
    n_params = [tf.reshape(n_params_raw[i], tf.shape(theta_G[i])) for i in range(Len_matrix)]
    N_params = [tf.reshape(N_params_raw[i], tf.shape(theta_G[i])) for i in range(Len_matrix)]

    # 影子生成器
    G_N_sample = generator_net_custom(X, M, N_params)
    G_n_sample = generator_net_custom(X, M, n_params)

    # 评估指标
    n_RMSE = tf.sqrt(tf.reduce_mean((M * X - M * G_n_sample) ** 2) / tf.reduce_mean(M))
    N_RMSE = tf.sqrt(tf.reduce_mean((M * X - M * G_N_sample) ** 2) / tf.reduce_mean(M))

    abs_diff = tf.abs(G_N_sample - G_n_sample)
    thre = tf.constant(value=thre_value, shape=[batch_size, dim])
    less = tf.reduce_sum(M * tf.cast(tf.less(abs_diff, thre), tf.float32))

    return {
        'ph': {'X': X, 'M': M, 'H': H, 'n_num': n_num},
        'ops': {'G_sample': G_sample, 'D_solver': D_solver, 'G_solver': G_solver, 'clip_d_op': clip_d_op},
        'losses': {'MSE': MSE_loss},
        'scis': {'N_RMSE': N_RMSE, 'n_RMSE': n_RMSE, 'less': less}
    }


def build_vaegain_graph(dim, params, gpu_device):
    """
    构建 VAE-GAIN 计算图
    """
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

    # Placeholders
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
    M = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
    H = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
    New_X = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])  # Input with noise/missing

    # Variables
    network_weights = vae_initialize_weights(dim,
                                             params['encoder_h1'], params['encoder_h2'],
                                             params['decoder_h1'], params['decoder_h2'],
                                             params['latent_size'])
    theta_D = vae_initialize_discriminator(dim, dim, dim)  # Using Dim as hidden size for D as in original

    # VAE Computation
    # 1. Encoder
    z_mean, z_log_sigma_sq = vae_recognition_network(New_X,
                                                     network_weights["weights_recog"],
                                                     network_weights["biases_recog"])

    # 2. Reparameterization Trick
    eps = tf.random.normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)
    z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

    # 3. Decoder (Generator)
    x_hat_mean, x_hat_log_sigma_sq = vae_generator_network(z,
                                                           network_weights["weights_gener"],
                                                           network_weights["biases_gener"])

    G_sample = x_hat_mean  # Use mean as the generated sample

    # 4. Reconstruction
    Hat_New_X = New_X * M + G_sample * (1 - M)

    # 5. Discriminator
    D_prob = vae_discriminator_network(Hat_New_X, H, theta_D)

    # Losses
    # D Loss
    D_loss = -tf.reduce_mean(M * tf.math.log(D_prob + 1e-8) + (1 - M) * tf.math.log(1. - D_prob + 1e-8))

    # VAE Loss
    # G_loss part from GAN
    G_loss_gan = -tf.reduce_mean((1 - M) * tf.math.log(D_prob + 1e-8))

    # KL Divergence & Reconstruction
    Normal = tf.compat.v1.distributions.Normal
    X_hat_distribution = Normal(loc=x_hat_mean, scale=tf.exp(x_hat_log_sigma_sq))

    # Original code: reconstr_loss = -tf.reduce_sum(M * X_hat_distribution.log_prob(New_X), 1)
    # Using compat.v1 log_prob
    reconstr_loss = -tf.reduce_sum(M * X_hat_distribution.log_prob(New_X), 1)
    latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)

    VAE_loss = G_loss_gan + params['alpha'] * tf.reduce_mean(reconstr_loss + latent_loss) / tf.reduce_mean(M)

    # Solvers
    D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    # G_solver (VAE Optimizer)
    # Original code uses RMSProp for VAE part
    VAE_solver = tf.compat.v1.train.RMSPropOptimizer(learning_rate=params['learning_rate']).minimize(VAE_loss)

    return {
        'ph': {'X': X, 'M': M, 'H': H, 'New_X': New_X},
        'ops': {'G_sample': G_sample, 'D_solver': D_solver, 'VAE_solver': VAE_solver},
        'losses': {'D_loss': D_loss, 'VAE_loss': VAE_loss}
    }