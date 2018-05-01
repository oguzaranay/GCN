import numpy as np
import random
import scipy.sparse as sp
import tensorflow as tf
from gcn.models import GCN
import time
from gcn.utils import construct_feed_dict


def get_data(split='train'):
    assert split in ['train', 'test']

    gsize = 10 # graph size, |V|
    sgsize = 5  # sub-graph size

    mu1, sigma1 = 1, 1.0 / gsize
    mu2, sigma2 = 5, 1.0 / sgsize

    adj = np.zeros((gsize, gsize)) # adjacency matrix

    if split == 'train':
        sample_size = 10
    else:
        sample_size = 2000

    features = np.zeros((sample_size, gsize))
    labels = np.zeros((sample_size, gsize))
    true_label = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
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
        idxs = random.sample(range(gsize), int(rho * gsize))
        s2 = [e for e in true_label]
        for j in idxs:
            s2[j] ^= 1
            if j in true_idx:
                s1[j] = np.random.normal(mu2, sigma2) # if node j belongs to true sub-graph then its feature is different

        features[i] = s1
        labels[i] = s2

    features = np.array(features, dtype=np.float)
    labels = np.array(labels, dtype=np.int)

    if split == 'train':
        return features, labels, true_label, adj
    else:
        return features, labels, true_label, adj

def f1_score(label,true_label):
    intersect = np.sum(np.min([label, true_label], axis=0))
    union = np.sum(np.max([label, true_label], axis=0))
    return 2 * intersect / float(intersect + max(10 ** -8, union))

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def train(features, labels, adj):
    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

    features = sp.lil_matrix(features)
    features = preprocess_features(features)
    support = [preprocess_adj(adj)]
    num_supports = 1
    idx_train = range(len(labels))
    train_mask = sample_mask(idx_train, labels.shape[0])

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = GCN(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    # Train model
    epochs = 200
    # early_stopping = 10
    for epoch in range(epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, labels, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: 0.5})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        # cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        # cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))
        # "val_loss=", "{:.5f}".format(cost), "val_acc=", "{:.5f}".format(acc),

        # if epoch > early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        #     print("Early stopping...")
        #     break

    print("Optimization Finished!")


if __name__ == '__main__':
    features, labels, true_label, adj = get_data(split='train')
    # f1 = f1_score(labels[0],true_label)
    # print("f1 = %.4f" % f1)
    # print(type(adj))
    train(features, labels, adj)