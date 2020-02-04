import scipy.sparse as sp
import numpy as np
import tensorflow as tf
import datetime
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

    def setup_logs(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        return train_summary_writer, test_summary_writer

    def fit(self, train_zip, val_zip, positive_weight=1, epochs=3):
        """
        Fits the model over `epochs` epochs (default 3).
        train_zip: tuple (feats, laplacians, targets, masks)
        val_zip: tuple (feats, laplacians, targets, masks)
        """
        # TODO: class balance
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
        train_accuracy = tf.keras.metrics.AUC()
        train_recall = tf.keras.metrics.Recall()
        train_precision = tf.keras.metrics.Precision()
        test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)
        test_accuracy = tf.keras.metrics.AUC()
        test_recall = tf.keras.metrics.Recall()
        test_precision = tf.keras.metrics.Precision()
        train_wr, test_wr = self.setup_logs()


        # Iterate over epochs.
        for epoch in range(epochs):

            print('Start of epoch %d' % (epoch,))

            # Iterate over the graphs.
            train_conf = None

            for step, (x, y, L, mask) in enumerate(zip(*train_zip)):
                batch_size = mask.shape[0]
                #rint(mask)
                #print(y)
                weighted_mask = tf.multiply(mask, y*positive_weight + (1-y))
                if step == 0 and epoch == 0: print(weighted_mask)
                with tf.GradientTape() as tape:
                    ỹ = self(x, L)
                    # loss requires (batch_size, num_classes)
                    loss = loss_fn(tf.reshape(y, [batch_size, 1]), ỹ, sample_weight=weighted_mask)

                ỹ = tf.reshape(ỹ, [-1])
                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))
                train_loss(loss)
                train_accuracy(y, ỹ, sample_weight=mask)
                train_recall(y, ỹ, sample_weight=mask)
                train_precision(y, ỹ, sample_weight=mask)
                pred_classes = tf.cast(tf.math.greater(ỹ, 0.5), tf.int32)
                if train_conf is None:
                    train_conf = tf.math.confusion_matrix(y, pred_classes, weights=mask)
                else:
                    train_conf = train_conf + tf.math.confusion_matrix(y, pred_classes, weights=mask)


            # Validation step.
            self.conf_matrix = None
            for x, y, L, mask in zip(*val_zip):
                ỹ = self(x, L)
                ỹ = tf.reshape(ỹ, [-1])
                loss = loss_fn(y, ỹ)
                test_loss(loss)
                test_accuracy(y, ỹ, sample_weight=mask)
                test_recall(y, ỹ, sample_weight=mask)
                test_precision(y, ỹ, sample_weight=mask)
                pred_classes = tf.cast(tf.math.greater(ỹ, 0.5), tf.int32)
                if self.conf_matrix is None:
                    self.conf_matrix = tf.math.confusion_matrix(y, pred_classes, weights=mask)
                else:
                    self.conf_matrix = self.conf_matrix + tf.math.confusion_matrix(y, pred_classes, weights=mask)

            with train_wr.as_default():
                tf.summary.scalar("loss", train_loss.result(), step=epoch)
                tf.summary.scalar("auc", train_accuracy.result(), step=epoch)
                tf.summary.scalar("recall", train_recall.result(), step=epoch)
                tf.summary.scalar("precision", train_precision.result(), step=epoch)
            with test_wr.as_default():
                tf.summary.scalar("loss", test_loss.result(), step=epoch)
                tf.summary.scalar("auc", test_accuracy.result(), step=epoch)
                tf.summary.scalar("recall", test_recall.result(), step=epoch)
                tf.summary.scalar("precision", test_precision.result(), step=epoch)

            print("Epoch {}:\n\tTRAIN loss {:.2f}, auc {:.2f}, recall {:.2f}, precision {:.2f}\n\tVAL loss {:.2f} auc {:.2f} recall {:.2f} precision {:.2f}".format(epoch,
                train_loss.result(), train_accuracy.result(), train_recall.result(), train_precision.result(),
                test_loss.result(), test_accuracy.result(), test_recall.result(), test_precision.result()))
            print("Confusion matrix(TRAIN):\n{}\nConfusion matrix(VAL):\n{}".format(
                train_conf, self.conf_matrix))

            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()