import scipy.sparse as sp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Laplacian:
    @staticmethod
    def from_adjacency(adj):
        adj = adj + sp.eye(adj.shape[0])
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


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


class LaplacianConvolution(layers.Layer):
    def __init__(self, output_dim, activation="relu", **kwargs):
        self.output_dim = output_dim
        if activation == "relu":
            self.activation = tf.keras.activations.relu
        elif activation == "sigmoid":
            self.activation = tf.keras.activations.sigmoid
        else:
            raise NotImplementedError
        super(LaplacianConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv_weights = self.add_weight(
            name='convweights', shape=(input_shape[1], self.output_dim),
            initializer='uniform', trainable=True)
        self.conv_bias = self.add_weight(
            name="convbias", shape=(self.output_dim),
            initializer="zeros", trainable=True
        )
        super(LaplacianConvolution, self).build(input_shape)

    def call(self, x, laplacian):
        #transformed = sparse_tensor_dense_matmul(x, self.weights)
        transformed = tf.matmul(x, self.conv_weights)
        masked = tf.sparse.sparse_dense_matmul(laplacian, transformed)
        return self.activation(masked + self.conv_bias)

class GraphConvolutionalNetwork(tf.keras.Model):
    def __init__(self, input_shape, output_dim, laplacian_shape, **kwargs):
        super(GraphConvolutionalNetwork, self).__init__(**kwargs)
        self.laplacian = tf.Variable#TODO
        self.conv1 = LaplacianConvolution(16, input_shape=input_shape)
        self.conv2 = LaplacianConvolution(output_dim, activation="sigmoid")

        for layer in self.layers:
            for weight in layer.weights:
                self.add_loss(tf.nn.l2_loss(weight))

    def call(self, x, laplacian):

        return self.conv2(self.conv1(x, laplacian), laplacian)

    def fit(self, feats, laplacians, targets, masks, epochs=3):
        # TODO: class balance
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        loss_metric = tf.keras.metrics.AUC()

        # Iterate over epochs.
        for epoch in range(epochs):
            print('Start of epoch %d' % (epoch,))

            # Iterate over the graphs.
            for step, (x, y, L, mask) in enumerate(zip(feats, targets, laplacians, masks)):
                with tf.GradientTape() as tape:
                    ỹ = self(x, L)
                    ỹ = tf.reshape(ỹ, [-1])
                    loss = loss_fn(y, ỹ)

                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                loss_metric(y, ỹ, sample_weight=mask)

                if step % 100 == 0:
                    print('step %s: mean loss = %s' % (step, loss_metric.result()))