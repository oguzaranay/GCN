import numpy as np
from numpy.linalg import inv
import random
import scipy.sparse as sp
import tensorflow as tf
import time
import matplotlib.pyplot as plt

n_nodes_hl1 = 16
n_nodes_hl2 = 20
n_classes = 10  # each class holds the probability of being a sub-graph node
batch_size = 1
tf.set_random_seed(123)

# sess = tf.InteractiveSession()


def preprocess_features(adj):
    # Normalize adjacency matrix
    # A_tilde
    A_tilde = adj + np.eye(n_classes) # A = A + I

    # D_tilde
    D_tilde = np.sum(A_tilde, axis=0) # row sum

    # A_hat
    D_inv_sqrt = np.power(D_tilde, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = sp.diags(D_inv_sqrt).toarray()
    # x1 = D_mat_inv_sqrt * D_mat_inv_sqrt * A_tilde
    x = np.matmul(np.matmul(D_mat_inv_sqrt, A_tilde), D_mat_inv_sqrt)

    # print(x1)
    # print(x)
    return x


def get_data(split='train'):
    assert split in ['train', 'test']

    gsize = 10 # graph size, |V|
    sgsize = 5  # sub-graph size

    mu1, sigma1 = 1, 1.0 / gsize
    mu2, sigma2 = 5, 1.0 / sgsize

    adj = np.zeros((gsize, gsize)) # adjacency matrix

    if split == 'train':
        sample_size = 1000
    else:
        sample_size = 1
        # true_label = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    features = np.zeros((sample_size, gsize))
    labels = np.zeros((sample_size, gsize))
    true_label = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    true_features = np.zeros(gsize, dtype='float64')
    for i in range(gsize):
        if true_label[i] == 0:
            true_features[i] = np.random.normal(mu1, sigma1)
        else:
            true_features[i] = np.random.normal(mu2, sigma2)
    # true_features = np.random.normal(mu1, sigma1, 5)
    # true_features = np.append(true_features, np.random.normal(mu2, sigma2, 5))

    true_idx = []
    for i in range(len(true_label)):
        if true_label[i] == 1:
            true_idx.append(i)

    # adjacency matrix for the graph of 10 nodes, V = {0, ..., 9}

    adj[0][1] = adj[1][0] = 1
    adj[0][2] = adj[2][0] = 1
    adj[0][3] = adj[3][0] = 1

    adj[1][3] = adj[3][1] = 1
    adj[1][4] = adj[4][1] = 1

    adj[2][3] = adj[3][2] = 1

    adj[3][4] = adj[4][3] = 1
    adj[3][7] = adj[7][3] = 1

    adj[5][6] = adj[6][5] = 1
    adj[5][8] = adj[8][5] = 1

    adj[6][7] = adj[7][6] = 1
    adj[6][8] = adj[8][6] = 1
    adj[6][9] = adj[9][6] = 1

    adj[7][9] = adj[9][7] = 1

    adj[8][9] = adj[9][8] = 1

    rho = 0.5  # controls number of nodes to be flipped in candidate label

    for i in range(sample_size):
        s1 = np.random.normal(mu1, sigma1, gsize) # assume all nodes don't belong to the true sub-graph
        # generate \hat(y)^{(i)}
        idxs = random.sample(range(gsize), int(rho * gsize)) # randomly select rho * |V| nodes
        s2 = [e for e in true_label] # make a copy from the true label
        for j in idxs:
            s2[j] ^= 1 # flips 0 to 1 and 1 to 0
            # if j in true_idx:
                # if node j belongs to true sub-graph then its feature is different
        for k in range(gsize):
            if s2[k] == 1:
                s1[k] = np.random.normal(mu2, sigma2)

        features[i] = s1
        labels[i] = s2

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # x = preprocess_features(features, adj)

    if split == 'train':
        return features, true_features, labels, true_label, adj, sample_size
    else:
        return true_features, true_label

def f1_score(label,true_label):
    intersect = np.sum(np.min([label, true_label], axis=0))
    union = np.sum(np.max([label, true_label], axis=0))
    return 2 * intersect / float(intersect + max(10 ** -8, union))

def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial)

# def multiply(adj, x):
#     # print(x[1])
#     out = np.zeros((10, 10), dtype=float)
#     for idx in range(10):
#         out[idx] = np.dot(adj, x[idx])

def neural_network(features):

    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([1, n_nodes_hl1], dtype='float64')),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1], dtype='float64'))}
    # hidden_layer_1 = {'weights':glorot([10, n_nodes_hl1]),
    #                   'biases':glorot([1, n_nodes_hl1])}

    # hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    #                   'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    #
    # output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_classes])),
    #                   'biases': tf.Variable(tf.random_normal([n_classes]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, 1], dtype='float64')),
                    'biases': tf.Variable(tf.random_normal([n_classes, 1], dtype='float64'))}
    # output_layer = {'weights': glorot([n_nodes_hl1, n_classes]),
    #                 'biases': glorot([1, n_classes])}

    # X = tilde(A) * features
    A_hat = preprocess_features(adj)

    # multiply features \in R^{sample_size x n} by normalized adj \in R^{n x n} = features \in R^{sample_size x n}
    features = tf.matmul(A_hat, features) # nx1

    l1 = tf.add(tf.matmul(features, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1) # 10x16

    # l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    # l2 = tf.nn.relu(l2)
    #
    # output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])

    l1 = tf.matmul(A_hat, l1) # 10x16
    output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])

    return output

def get_batch(features, labels, offset):
    x = features[offset:offset + batch_size, :]
    y = labels[offset:offset + batch_size, :]
    return x, y

def project(input):
    x = np.array(input)
    output = [1 if item >= 0.5 else 0 for item in x]
    return output

def train(features, adj, labels, sample_size):

    x = tf.placeholder('float64', [n_classes, 1], name='X')
    y = tf.placeholder('float64', [n_classes, 1], name='Y')

    lr = 0.001
    raw_prediction = neural_network(x)
    # prediction = project(prediction)
    prediction = tf.nn.sigmoid(raw_prediction, name='Prediction')
    # cost = tf.metrics.mean_iou(labels=y, predictions=prediction, num_classes=10, name='Mean_IOU')
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=raw_prediction, name='Sigmoid'))
    # cost = tf.reduce_mean(tf.metrics.precision(labels=y, predictions=prediction, name='F1_Score'))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    epochs = 10
    losses = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # start training
        for epoch in range(epochs):
            loss = 0
            for i in range(0, sample_size, batch_size):
                x_input, y_label = get_batch(features, labels, i)
                # sess.run(tf.local_variables_initializer())
                _, c = sess.run([optimizer, cost], feed_dict={x: x_input.reshape([n_classes, 1]), y: y_label.reshape([n_classes, 1])})
                loss += c
            loss /= (sample_size / batch_size)
            print('Epoch:', epoch, 'completed out of:', epochs, 'loss:', loss)
            losses.append(loss)
        # sess.run(tf.global_variables_initializer())

        # start testing
        x_test, gt = get_data(split='test')

        correct = tf.equal(tf.round(prediction), y)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))

        gt = np.array(gt).reshape([1, -1])
        x_test = np.array(x_test).reshape([1, -1])
        c_test, acc, pred = sess.run([cost, accuracy, prediction], feed_dict={x: x_test.reshape([n_classes, 1]), y: gt.reshape([n_classes, 1])})
        print('Net Accuracy: %.4f' % acc)
        print('Test cost: %.4f' % c_test)
        print('Test input:', np.round(x_test, 3))
        print('Predicted label   :', np.round(pred.reshape([1, n_classes])).astype(int))
        print('Ground truth label:', gt)

    plt.plot(losses, 'r')
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['loss'])
    plt.title('Training Neural Network')
    plt.text(epochs * .5, np.max(losses) * .5, '$H = 1$')
    plt.text(epochs * .5, np.max(losses) * .45, '$C^{(1)} = $' + str(n_nodes_hl1))
    plt.text(epochs * .5, np.max(losses) * .4, '$\eta = $' + str(lr))
    plt.text(epochs * .5, np.max(losses) * .35, '$K = $' + str(sample_size))
    plt.text(epochs * .5, np.max(losses) * .3, '$Net\ accuracy = $' + str(acc))
    plt.text(epochs * .5, np.max(losses) * .25, '$Test\ cost = $' + str(np.round(c_test, 4)))
    plt.text(epochs * .5, np.max(losses) * .2, '$w^{(0)},\ w^{(1)}=N(0,1)$')
    plt.axis([0, epochs - 1, 0, np.max(losses)])

    plt.show()

features, true_features, labels, true_label, adj, sample_size = get_data(split='train')
# features = tf.convert_to_tensor(features,dtype='float')
# labels = tf.convert_to_tensor(labels, dtype='float')
# true_label = tf.convert_to_tensor(true_label)

train(features, adj, labels, sample_size)