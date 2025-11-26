#%% Packages
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow.compat.v1 as tf
from Algorithm.utils import  rounding

tf.disable_v2_behavior()
gpu_options = tf.GPUOptions(allow_growth=True)
Normal = tf.distributions.Normal

# normalize the input imputing dataset
def normalization(data, parameters=None):
    '''Normalize data in [0, 1] range.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization
    '''

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:

        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

            # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.

    Args:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization

    Returns:
      - renorm_data: renormalized original data
    '''

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]

    return renorm_data


def VAE_GAIN(df_from_system):
    start = time.clock()
    df = df_from_system
    all_data_original = np.array(df)
    df = pd.concat([df.columns.to_frame().T, df], ignore_index=True)
    df = df.astype(str)
    df = df.values
    df[df == 'nan'] = ''
    #print(df)
    o_data = df
    #print(o_data)
    headers = o_data[0, :].tolist()  # get headers in line 0
    o_data = o_data[1:, :]
    o_data[o_data == ""] = np.nan
    o_data = o_data.astype(float)
    No, Dim = o_data.shape
    Missing = 1 - np.isnan(o_data)  # miss mask
    miss = No * Dim - np.count_nonzero(Missing)

    norm_data, norm_parameters = normalization(o_data)

    # %% System Parameters
    # 1. Mini batch size
    mb_size = 8
    # 2. Missing rate
    p_miss = miss / (Dim * No)
    # 3. Hint rate
    p_hint = 0.9
    # 4. Loss Hyperparameters
    alpha = 10
    # 5. Train Rate
    train_rate = 1
    # 6. Epoch Number
    Epoch_number = 1
    # 7. discriminator_number
    discriminator_number = 1
    generator_number = 1
    # %% Data
    p_miss_list = [p_miss]

    Data = np.nan_to_num(norm_data, 0)
    train_list = Data
    Method = 'VAEGAIN'
    epoch = 0

    H_Dim1 = Dim
    H_Dim2 = Dim

    relationship = np.corrcoef(Data, rowvar=False) * 0.5 + 0.5
    relationship = relationship[-1, :]
    rank = np.argsort(-relationship, axis=0)[1:]
    Top_K = int((0.5 + 0.5 * p_miss) * Dim)
    # Top_feature = rank[:Top_K]
    Bottom = rank[Top_K:]

    idx = np.random.permutation(No)  # Random Index

    Train_No = len(Data)  # The number of tuples in training set
    # Test_No = len(test_list)  # The number of tuples in test set

    trainX = Data  # total dataset is train set
    # testX = Data[test_list, :]

    # Train / Test Missing Indicators
    trainM = Missing

    # %% Necessary Functions

    # 1. Xavier Initialization Definition
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def xavier_init_AE(fan_in, fan_out, constant=1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),
                                 minval=low, maxval=high,
                                 dtype=tf.float32)

    # Hint Vector Generation
    def sample_M(m, n, p):
        A = np.random.uniform(0., 1., size=[m, n])
        B = A > p
        C = 1. * B
        return C

    '''
    GAIN Consists of 3 Components
    - Generator
    - Discriminator
    - Hint Mechanism
    '''
    # start = time.clock()
    # %% GAIN Architecture
    Encoder_hidden1 = 50
    Encoder_hidden2 = 20
    Decoder_hidden1 = 50
    Decoder_hidden2 = 20
    latent_size = 20
    learning_rate = 0.002

    # %% 1. Input Placeholders
    # 1.1. Data Vector
    X = tf.placeholder(tf.float32, shape=[None, Dim])
    # 1.2. Mask Vector
    M = tf.placeholder(tf.float32, shape=[None, Dim])
    # 1.3. Hint vector
    H = tf.placeholder(tf.float32, shape=[None, Dim])
    # 1.4. X with missing values
    New_X = tf.placeholder(tf.float32, shape=[None, Dim])

    # %% 2. Discriminator
    D_W1 = tf.Variable(xavier_init([Dim * 2, H_Dim1]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

    D_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
    D_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))

    D_W3 = tf.Variable(xavier_init([H_Dim2, Dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[Dim]))  # Output is multi-variate

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # %% 3. Generator
    G_W1 = tf.Variable(
        xavier_init([Dim * 2, H_Dim1]))  # Data + Mask as inputs (Random Noises are in Missing Components)
    G_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

    G_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
    G_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))

    G_W3 = tf.Variable(xavier_init([H_Dim2, Dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[Dim]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    # GAIN Function

    network_architecture = \
        dict(n_hidden_recog_1=Encoder_hidden1,  # 1st layer encoder neurons
             n_hidden_recog_2=Encoder_hidden2,  # 2nd layer encoder neurons
             n_hidden_gener_1=Decoder_hidden1,  # 1st layer decoder neurons
             n_hidden_gener_2=Decoder_hidden2,  # 2nd layer decoder neurons
             n_input=Dim,  # data input size
             n_z=latent_size)  # dimensionality of latent space

    def _initialize_weights(n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
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

    def _recognition_network(weights, biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(New_X, weights['h1']),
                                    biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']),
                                    biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(z, weights, biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(z, weights['h1']),
                                    biases['b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']),
                                    biases['b2']))
        x_hat_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                            biases['out_mean'])
        x_hat_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (x_hat_mean, x_hat_log_sigma_sq)

    # Initialize autoencode network weights and biases
    network_weights = _initialize_weights(**network_architecture)

    # Use recognition network to determine mean and
    # (log) variance of Gaussian distribution in latent
    # space
    z_mean, z_log_sigma_sq = \
        _recognition_network(network_weights["weights_recog"],
                             network_weights["biases_recog"])
    eps = tf.random_normal(tf.shape(z_mean), 0, 1,
                           dtype=tf.float32)
    z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

    x_hat_mean, x_hat_log_sigma_sq = \
        _generator_network(z, network_weights["weights_gener"], network_weights["biases_gener"])

    # # 1. Generator
    # def generator(new_x, m):
    #     inputs = tf.concat(axis=1, values=[new_x, m])  # Mask + Data Concatenate
    #     G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    #     G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    #     G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)  # [0,1] normalized Output
    #
    #     return G_prob

    # 2. Discriminator
    def discriminator(new_x, h):
        inputs = tf.concat(axis=1, values=[new_x, h])  # Hint + Data Concatenate
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output

        return D_prob

    # 3. Other functions
    # Random sample generator for Z
    def sample_Z(m, n):
        return np.random.uniform(0., 0.01, size=[m, n])

    # Mini-batch generation
    def sample_idx(m, n):
        A = np.random.permutation(m)
        idx = A[:n]
        return idx

    # %% Structure
    # Generator

    G_sample, x_hat_log_sigma_sq_ = _generator_network(z, network_weights["weights_gener"],
                                                       network_weights[
                                                           "biases_gener"])  # Definition of generator: input X without missing values and Mask Vector

    # Combine with original data
    Hat_New_X = New_X * M + G_sample * (1 - M)  # Generated complete data

    # Discriminator
    # D_prob_G = discriminator(Hat_New_X, H)         # Definition of generator: input generated data and Hint Vector
    D_prob = discriminator(Hat_New_X, H)  # Definition of generator: input generated data and Hint Vector

    # %% Loss
    D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8))
    G_loss1 = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
    # MSE_train_loss = tf.reduce_mean((M * New_X - M * G_sample)**2) / tf.reduce_mean(M)

    X_hat_distribution = Normal(loc=x_hat_mean, scale=tf.exp(x_hat_log_sigma_sq))
    reconstr_loss = -tf.reduce_sum(M * X_hat_distribution.log_prob(New_X), 1)
    latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)

    D_loss = D_loss1
    # G_loss = G_loss1 + alpha * MSE_train_loss
    VAE_loss = G_loss1 + alpha * tf.reduce_mean(reconstr_loss + latent_loss) / tf.reduce_mean(M)  # average over batch

    # %% MSE Performance metric
    MSE_test_loss = tf.reduce_mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / tf.reduce_mean(1 - M)
    MAE_test_loss = tf.reduce_mean((((1 - M) * X - (1 - M) * G_sample) ** 2) ** 0.5) / tf.reduce_mean(1 - M)

    # %% Solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)  # Optimizer for D
    # G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)   # Optimizer for G
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(VAE_loss)

    # Sessions Definition
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    # %% Start Iterations
    # Loss_name = "test1_MCAR_missing_rate_" + str(int(p_miss * 100)) + '_Epoch' + str(epoch + 1) + '_' + Method + '_Loss.csv'
    # Loss = [['Iteration', 'D_loss', 'G_loss', 'MSE_train_loss']]

    for it in tqdm(range(5000)):
        Loss_each = []
        # %% Inputs
        mb_idx = sample_idx(len(Data), mb_size)  # Mini batch
        X_mb = Data[mb_idx, :]

        Z_mb = sample_Z(mb_size, Dim)
        M_mb = Missing[mb_idx, :]
        H_mb1 = sample_M(mb_size, Dim, 1 - p_hint)
        H_mb = M_mb * H_mb1

        New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce
        # Training discriminator
        for num_ in range(discriminator_number):
            _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={M: M_mb, New_X: New_X_mb, H: H_mb})
        for num_ in range(generator_number):
            _, G_loss_curr = sess.run([optimizer, VAE_loss], feed_dict={M: M_mb, New_X: New_X_mb, H: H_mb})

    # df = pd.DataFrame(Loss)
    # df.to_csv(Loss_name, index=False, header=False)
    # %% Final Loss
    # end = time.clock()
    # time_ = end - start

    Z_mb = sample_Z(len(Data), Dim)
    M_mb = Missing
    X_mb = Data

    New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce
    Data_Z_mb = sample_Z(len(Data), len(Data[0]))
    New_Data_mb = M_mb * X_mb + (1 - M_mb) * Data_Z_mb  # Missing Data Introduce

    MSE_final, MAE_final, Sample_ALL = sess.run([MSE_test_loss, MAE_test_loss, G_sample],
                                                feed_dict={X: X_mb, M: M_mb, New_X: New_Data_mb})
    predict_name = 'MCAR_missing_rate_' + str(int(p_miss * 100)) + '_' + Method + '_imputed.csv'
    # predict_name = "imputed_out.csv"
    imputed_data = Missing * norm_data + (1 - Missing) * Sample_ALL
    imputed_data = renormalization(imputed_data, norm_parameters)
    imputed_data = rounding(imputed_data, all_data_original)
    # imputed_data = np.around(imputed_data, 6)
    # imputed_data = pd.DataFrame(imputed_data, columns=headers)
    end = time.clock()
    print('总用时：', end - start)

    return imputed_data

