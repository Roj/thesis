import scipy.sparse as sp
import numpy as np
import datetime
import logging
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf
    from tensorflow.keras import layers

import sklearn.model_selection
from neighborhooditerator import NeighborhoodIterator
import progressbar

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

class SimpleLaplacian:
    @staticmethod
    def from_adjacency(adj):
        adj = adj + sp.eye(adj.shape[0])
        D = sp.diags(np.array(adj.sum(1)).flatten())
        return (D - adj).tocoo()

class RandomWalkLaplacian:
    @staticmethod
    def from_adjacency(adj, step = 1):
        adj = adj + sp.eye(adj.shape[0])
        # Row-normalize adjacency matrix
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        return adj.dot(d_mat_inv).transpose().tocoo()


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
        logging.info("Building with input_shape %s", input_shape)
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

class NormalizedLaplacianConvolution(LaplacianConvolution):
    def call(self, x, laplacian, node_mask):
        """Use batch normalization on the present nodes using node_mask
        and then calculate the convolution."""

        only_present_nodes = tf.boolean_mask(x, tf.math.equal(node_mask, 1))
        means = tf.expand_dims(tf.math.reduce_mean(only_present_nodes, axis=0), 0)
        stds = tf.expand_dims(tf.math.reduce_std(only_present_nodes, axis=0), 0)
        node_mask_expanded = tf.expand_dims(node_mask, 1)

        # Normalize each element by subtracting its column mean, dividing by its column
        # stdev and multiplying by its row mask.
        normalized = tf.nn.batch_normalization(
            x,
            means,
            stds,
            None,
            node_mask_expanded,
            variance_epsilon=0.001
        )

        # And now just proceed as a laplacian layer:
        return super().call(normalized, laplacian)


class GraphConvolutionalNetwork(tf.keras.Model):
    default_hyperparams = {
        "num_layers": 1,
        "learning_rate": 1e-3,
        "num_filters": 128,
        "batch_normalization": False
    }
    def __init__(self, input_shape, output_dim, laplacian_shape,
                 log_hyperparams=False, hyperparams=default_hyperparams, **kwargs):
        """Construct the GCN model.
        Parameters
        ----------
        input_shape: integer
            Number of features.
        output_dim: integer
            Number of classes to predict. Currently only 1 is supported, as we use binary cross entropy.
        laplacian_shape: iterable of integer
            Numpy shape of the laplacian.
        log_hyperparams: bool
            If True, use tensorboard to log hyperparameters. Default False.
        hyperparams: dict
            Specify num_layers, learning_rate, num_filters, batch_normalization. See
            GraphConvolutionalNetwork.defauly_hyperparams for more info.
        kwargs: dict
            Other special parameters, such as network name (`name`).

        """
        super(GraphConvolutionalNetwork, self).__init__(**kwargs)

        # If hyperparams is missing some keys, use default values.
        for key in self.default_hyperparams.keys():
            hyperparams.setdefault(key, self.default_hyperparams[key])

        self.hyperparams = hyperparams
        self.laplacian = tf.Variable
        self.laplacian_shape = laplacian_shape

        # Set up layers
        self.conv_layers = []
        hidden_dim = hyperparams["num_filters"]
        layer = LaplacianConvolution
        if hyperparams["batch_normalization"]:
            layer = NormalizedLaplacianConvolution

        for i in range(hyperparams["num_layers"]):
            if i == 0:
                self.conv_layers.append(layer(hidden_dim, input_shape=input_shape))
            else:
                self.conv_layers.append(layer(hidden_dim))

        self.conv_layers.append(layer(output_dim, activation="sigmoid"))
        self.network_name = kwargs.get("name", "")

        for layer in self.layers:
            for weight in layer.weights:
                self.add_loss(tf.nn.l2_loss(weight))

    def call(self, x, laplacian, mask):
        value = x
        for layer in self.conv_layers:
            if not self.hyperparams["batch_normalization"]:
                value = layer(value, laplacian)
            else:
                value = layer(value, laplacian, mask)

        return value

    def make_logs(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = f"logs/gradient_tape/{self.network_name}_{current_time}/train"
        test_log_dir = f"logs/gradient_tape/{self.network_name}_{current_time}/test"

        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        return train_summary_writer, test_summary_writer

    def setup_metrics(self, name):
        loss = tf.keras.metrics.Mean(f"{name}_loss", dtype=tf.float32)
        accuracy = tf.keras.metrics.AUC()
        recall = tf.keras.metrics.Recall()
        precision = tf.keras.metrics.Precision()
        return loss, accuracy, recall, precision

    def log_metrics(self, loss, acc, recall, prec, epoch):
        tf.summary.scalar("loss", loss.result(), step=epoch)
        tf.summary.scalar("auc", acc.result(), step=epoch)
        tf.summary.scalar("recall", recall.result(), step=epoch)
        tf.summary.scalar("precision", prec.result(), step=epoch)

    def reset_metrics(self, loss, acc, recall, prec):
        loss.reset_states()
        acc.reset_states()
        recall.reset_states()
        prec.reset_states()

    def update_conf_matrix(self, y, ỹ, mask, conf):
        pred_classes = tf.cast(tf.math.greater(ỹ, 0.5), tf.int32)
        if conf is None:
            return tf.math.confusion_matrix(y, pred_classes, weights=mask)

        return  conf + tf.math.confusion_matrix(y, pred_classes, weights=mask)

    def fit_cv_groups(self, data_zip, groups, folds=5, positive_weight=1.0, epochs=3, restore_best=False):
        """
        Fits the model using group cross validation over `epochs` epochs
        (default 3).
        Parameters
        ----------
        data_zip: tuple
            List of (feats, targets, laplacians, masks)
        groups: iterable
            Group for each
        folds: int
            Number of folds to use. Default 5.
        positive_weight: float
            Weight of positive class. Default 1.
        epochs: integer
            Number of epochs to fit each fold for. Default 3.
        restore_best: bool
            Whether to restore the best score weights after fitting all folds.
            Default False.

        Returns
        ------
        score: float
            mean of each validation fold AUC
        """
        group_kfold = sklearn.model_selection.GroupKFold(folds)
        feats, laplacians, masks = data_zip[0], data_zip[2], data_zip[3]
        # So far the weights haven't been initialized, and
        # saving weights will save 0 layers. We can force
        # initialization like so:
        logging.info("Initializing GCN with %s, %s, %s", feats[0], laplacians[0], masks[0])
        _ = self(feats[0], laplacians[0], masks[0])
        # And now we can save the starting point
        logging.info("And saving starting point.")
        self.save_weights('init.h5')
        best_score = 0
        all_scores = []
        original_name = self.network_name
        self.network_name += "0"
        for fold, (train_idx, test_idx) in enumerate(group_kfold.split(feats, groups=groups)):
            logging.info("Resetting weights..")
            self.network_name = self.network_name[:-1] + str(fold)
            self.load_weights('init.h5')

            train_zip = [
                [seq[i] for i in train_idx]
                for seq in data_zip
            ]
            test_zip = [
                [seq[i] for i in test_idx]
                for seq in data_zip
            ]
            logging.info("Fitting fold %s", fold)
            score = self.fit(
                train_zip,
                test_zip,
                positive_weight=positive_weight,
                epochs=epochs,
                verbose=False
            )
            all_scores.append(score)
            if score > best_score:
                logging.info("Best score achieved, saving weights..")
                self.save_weights(f"best_{original_name}.h5")
                best_score = score

        logging.info(f"Best score is {best_score}, loading weights saved in best_{original_name}.h5")
        self.load_weights(f"best_{original_name}.h5")

        return sum(all_scores)/folds


    def fit(self, train_zip, val_zip, positive_weight=1.0, epochs=3, verbose=False):
        """
        Fits the model over `epochs` epochs (default 3).
        Parameters
        ----------
        train_zip: tuple
            Tuple of (feats, targets, laplacians,  masks)
        val_zip: tuple
            same as train_zip
        positive_weight: float (default 1.0)
            weight for positive class
        epochs: integer (default 3)
        verbose: bool (default False)
        Returns:
        auc: float
            best auc achieved by the model
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.hyperparams["learning_rate"])
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        train_loss, train_accuracy, train_recall, train_precision = self.setup_metrics("train")
        test_loss, test_accuracy, test_recall, test_precision = self.setup_metrics("test")
        train_wr, test_wr = self.make_logs()

        # Iterate over epochs.
        best_auc = 0
        for epoch in range(epochs):

            logging.info('Start of epoch %d' % (epoch,))
            # Iterate over the graphs.
            train_conf = None

            for step, (x, y, L, mask) in enumerate(zip(*train_zip)):
                batch_size = mask.shape[0]
                weighted_mask = tf.multiply(mask, y*positive_weight + (1-y))
                if step == epoch == 0 and verbose: logging.info(weighted_mask)

                with tf.GradientTape() as tape:
                    ỹ = self(x, L, mask)
                    # loss requires (batch_size, num_classes)
                    loss = loss_fn(tf.reshape(y, [batch_size, 1]), ỹ, sample_weight=weighted_mask)

                ỹ = tf.reshape(ỹ, [-1])

                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                train_loss(loss)
                train_accuracy(y, ỹ, sample_weight=mask)
                train_recall(y, ỹ, sample_weight=mask)
                train_precision(y, ỹ, sample_weight=mask)
                train_conf = self.update_conf_matrix(y, ỹ, mask, train_conf)

            # Validation step.
            test_conf = None
            for step, (x, y, L, mask) in enumerate(zip(*val_zip)):
                batch_size = mask.shape[0]
                weighted_mask = tf.multiply(mask, y*positive_weight + (1-y))
                ỹ = self(x, L, mask)
                loss = loss_fn(tf.reshape(y, [batch_size, 1]), ỹ, sample_weight=weighted_mask)
                ỹ = tf.reshape(ỹ, [-1])

                test_loss(loss)
                test_accuracy(y, ỹ, sample_weight=mask)
                test_recall(y, ỹ, sample_weight=mask)
                test_precision(y, ỹ, sample_weight=mask)
                test_conf = self.update_conf_matrix(y, ỹ, mask, test_conf)

            with train_wr.as_default():
                self.log_metrics(train_loss, train_accuracy, train_recall, train_precision, epoch)

            with test_wr.as_default():
                self.log_metrics(test_loss, test_accuracy, test_recall, test_precision, epoch)

            logging.info("Epoch {}:\n\tTRAIN loss {:.2f}, auc {:.2f}, recall {:.2f}, precision {:.2f}\n\tVAL loss {:.2f} auc {:.2f} recall {:.2f} precision {:.2f}".format(epoch,
                train_loss.result(), train_accuracy.result(), train_recall.result(), train_precision.result(),
                test_loss.result(), test_accuracy.result(), test_recall.result(), test_precision.result()))
            logging.info("Confusion matrix(TRAIN):\n{}\nConfusion matrix(VAL):\n{}".format(
                train_conf, test_conf))

            best_auc = max(best_auc, test_accuracy.result())
            self.reset_metrics(train_loss, train_accuracy, train_recall, train_precision)
            self.reset_metrics(test_loss, test_accuracy, test_recall, test_precision)
        return best_auc


class LocalGCN(GraphConvolutionalNetwork):
    def __init__(self, *args, **kwargs):
        super(LocalGCN, self).__init__(*args, **kwargs)
    def fit(self, train_zip, val_zip, positive_weight=1.0, epochs=3, verbose=False):
        """
        Fits the model over `epochs` epochs (default 3).
        Parameters
        ----------
        train_zip: tuple
            Tuple of (feats, targets, laplacians,  masks, last_neighborhood)
        val_zip: tuple
            same as train_zip
        positive_weight: float (default 1.0)
            weight for positive class
        epochs: integer (default 3)
        verbose: bool (default False)
        Returns:
        auc: float
            best auc achieved by the model
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        train_loss, train_accuracy, train_recall, train_precision = self.setup_metrics("train")
        test_loss, test_accuracy, test_recall, test_precision = self.setup_metrics("test")
        train_wr, test_wr = self.make_logs()

        # Iterate over epochs.
        best_auc = 0
        for epoch in range(epochs):

            logging.info('Start of epoch %d' % (epoch,))
            # Iterate over the graphs.
            train_conf = None
            it = progressbar.ProgressBar()(enumerate(zip(*train_zip)), max_value=len(train_zip[0]))
            for step, (x, y, L, mask, last_neigh) in it:

                with tf.GradientTape() as tape:
                    weighted_masks_concatenated = []
                    masks_concatenated = []
                    ỹ_concatenated = []
                    y_concatenated = []
                    while last_neigh == 0:
                        weighted_mask = tf.multiply(mask, y*positive_weight + (1-y))
                        if step == epoch == 0 and verbose: logging.info(weighted_mask)
                        ỹ = self(x, L, mask)
                        masks_concatenated.append(mask)
                        weighted_masks_concatenated.append(weighted_mask)
                        ỹ_concatenated.append(ỹ)
                        y_concatenated.append(y)
                        step, (x, y, L, mask, last_neigh) = next(it)

                    mask = tf.concat(masks_concatenated, axis=0)
                    weighted_mask = tf.concat(weighted_masks_concatenated, axis=0)
                    ỹ = tf.concat(ỹ_concatenated, axis=0)
                    y = tf.concat(y_concatenated, axis=0)
                    # loss requires (batch_size, num_classes)
                    loss = loss_fn(
                        tf.reshape(y, [weighted_mask.shape[0], 1]),
                        ỹ,
                        sample_weight=weighted_mask)

                ỹ = tf.reshape(ỹ, [-1])

                grads = tape.gradient(loss, self.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                train_loss(loss)
                train_accuracy(y, ỹ, sample_weight=mask)
                train_recall(y, ỹ, sample_weight=mask)
                train_precision(y, ỹ, sample_weight=mask)
                train_conf = self.update_conf_matrix(y, ỹ, mask, train_conf)

            # Validation step.
            test_conf = None
            it = enumerate(zip(*val_zip))
            for step, (x, y, L, mask, last_neigh) in it:
                weighted_masks_concatenated = []
                masks_concatenated = []
                ỹ_concatenated = []
                y_concatenated = []
                while last_neigh == 0:
                    weighted_mask = tf.multiply(mask, y*positive_weight + (1-y))
                    ỹ = self(x, L, mask)
                    masks_concatenated.append(mask)
                    weighted_masks_concatenated.append(weighted_mask)
                    ỹ_concatenated.append(ỹ)
                    y_concatenated.append(y)
                    step, (x, y, L, mask, last_neigh) = next(it)

                mask = tf.concat(masks_concatenated, axis=0)
                weighted_mask = tf.concat(weighted_masks_concatenated, axis=0)
                ỹ = tf.concat(ỹ_concatenated, axis=0)
                y = tf.concat(y_concatenated, axis=0)
                loss = loss_fn(
                        tf.reshape(y, [weighted_mask.shape[0], 1]),
                        ỹ,
                        sample_weight=weighted_mask)

                ỹ = tf.reshape(ỹ, [-1])

                test_loss(loss)
                test_accuracy(y, ỹ, sample_weight=mask)
                test_recall(y, ỹ, sample_weight=mask)
                test_precision(y, ỹ, sample_weight=mask)
                test_conf = self.update_conf_matrix(y, ỹ, mask, test_conf)

            with train_wr.as_default():
                self.log_metrics(train_loss, train_accuracy, train_recall, train_precision, epoch)

            with test_wr.as_default():
                self.log_metrics(test_loss, test_accuracy, test_recall, test_precision, epoch)

            logging.info("Epoch {}:\n\tTRAIN loss {:.2f}, auc {:.2f}, recall {:.2f}, precision {:.2f}\n\tVAL loss {:.2f} auc {:.2f} recall {:.2f} precision {:.2f}".format(epoch,
                train_loss.result(), train_accuracy.result(), train_recall.result(), train_precision.result(),
                test_loss.result(), test_accuracy.result(), test_recall.result(), test_precision.result()))
            logging.info("Confusion matrix(TRAIN):\n{}\nConfusion matrix(VAL):\n{}".format(
                train_conf, test_conf))

            best_auc = max(best_auc, test_accuracy.result())
            self.reset_metrics(train_loss, train_accuracy, train_recall, train_precision)
            self.reset_metrics(test_loss, test_accuracy, test_recall, test_precision)
        return best_auc
