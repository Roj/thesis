import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from gcn.gcn import GraphConvolutionalNetwork
import logging

class HyperparameterDetective:
    def __init__(self, run_dir_prefix, experiment_name, hparams_domains,
                 input_shape, output_dim, laplacian_shape, **kwargs):

        self.run_dir_prefix = run_dir_prefix
        self.experiment_name = experiment_name
        self.hparams_domains = self.prepare_hparams(hparams_domains)

        def create_model(hparams):
            model = GraphConvolutionalNetwork(
                input_shape, output_dim, laplacian_shape, log_hyperparams=True,
                hyperparams=hparams, **kwargs
            )
            return model

        self.create_model = create_model

    def prepare_hparams(self, hparams_domains):
        """Convert informal specifications to a dict of hp.HParam"""
        domains = {}
        domains["num_layers"] = hp.HParam(
            "num_layers",
            hp.Discrete(hparams_domains["num_layers"]))
        domains["learning_rate"] = hp.HParam(
            "learning_rate",
            hp.RealInterval(*hparams_domains["learning_rate"]))
        domains["num_filters"] = hp.HParam(
            "num_filters",
            hp.Discrete(hparams_domains["num_filters"]))
        domains["batch_normalization"] = hp.HParam(
            "batch_normalization",
            hp.Discrete(hparams_domains["batch_normalization"]))
        return domains

    def run_log_cv(self, hparams, num,
                   data_zip, groups, folds, positive_weight, epochs):
        """Run one instance of model (using CV) and log results"""
        logging.info("[HyperparameterDetective] Running with params %s", hparams)
        with tf.summary.create_file_writer(f"{self.run_dir_prefix}/{self.experiment_name}_{num}").as_default():
            hp.hparams(hparams)  # record the values used in this trial
            avg_auc = self.create_model(hparams).fit_cv_groups(
                data_zip, groups, folds, positive_weight, epochs, False)
            tf.summary.scalar("auc", avg_auc, step=1)
        logging.info("[HyperparameterDetective] AVG AUC = %s", avg_auc)
        return avg_auc

    def get_hparams_combinations(self, lr_sample_amount=1):
        for layers in self.hparams_domains["num_layers"].domain.values:
            for i in range(lr_sample_amount):
                for batch_norm in self.hparams_domains["batch_normalization"].domain.values:
                    #learning_rate=self.hparams_domains["learning_rate"].domain.sample_uniform()
                    learning_rate=1e-3
                    for filters in self.hparams_domains["num_filters"].domain.values:
                        yield {
                            "num_layers": layers,
                            "learning_rate": learning_rate,
                            "num_filters": filters,
                            "batch_normalization": batch_norm
                        }

    def search(self, data_zip, groups, folds=5, positive_weight=1.0, epochs=40):
        hparam_tuning_fn = f"{self.run_dir_prefix}/hparam_tuning"
        with tf.summary.create_file_writer(hparam_tuning_fn).as_default():
            hp.hparams_config(
                hparams=list(self.hparams_domains.values()),
                metrics=[hp.Metric("auc", display_name='Average AUC')]
            )
        num = 0
        best_auc = 0
        best_params = None
        for run_num, hparam_run in enumerate(self.get_hparams_combinations()):
            auc = self.run_log_cv(hparam_run, run_num,
                            data_zip, groups, folds, positive_weight, epochs)
            if auc > best_auc:
                best_params = hparam_run
                best_auc = auc

        logging.info("[Hyperparameter Detective] Best AVG AUC is %s with hparams %s", best_auc, best_params)