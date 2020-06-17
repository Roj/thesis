import yaml
import logging
import pathlib
from thesispipeline import ThesisPipeline

class ExperimentExecutor:
    def run_gcn(self, contacts=False, neighborhoods=False,
                normalize_adj=False, epochs=2, filter_node_amount=None,
                dist=3, negative_prob=0.5):
        thesis = ThesisPipeline()

        if contacts:
            thesis.load_data(graphs_folder="graphs/contacts/", names_groups_file="names_groups.pkl")
            thesis.filter_graphs_missing_data()
        else:
            thesis.load_data(graphs_folder="graphs/", names_groups_file="names_groups.pkl")

        thesis.make_protein_groups()

        thesis.check_protein_groups()

        if not contacts:
            thesis.remove_interior_nodes()

        thesis.make_features()
        thesis.make_adjacency_matrices()

        if normalize_adj:
            thesis.normalize_adjacency_matrices()

        thesis.make_targets()
        thesis.convert_matrices_32bits()

        if filter_node_amount is not None:
            thesis.filter_by_node_amount(amount=filter_node_amount)

        if neighborhoods:
            thesis.make_neighborhoods(dist=dist, negative_prob=negative_prob, verbose=1)

        thesis.make_positive_weight()
        thesis.make_laplacians()
        thesis.pad_matrices()

        if neighborhoods:
            thesis.make_masks(pad_only=True)
            thesis.run_cv_local_gcn(epochs=epochs, name=self.experiment_name)
        else:
            thesis.make_masks()
            thesis.run_cv_gcn(epochs=epochs, name=self.experiment_name)

    def run_xgb(self, contacts=False, steps=3, filter_node_amount=None):
        thesis = ThesisPipeline()
        if contacts:
            thesis.load_data(graphs_folder="graphs/contacts/", names_groups_file="names_groups.pkl")
            thesis.filter_graphs_missing_data()
        else:
            thesis.load_data(graphs_folder="graphs/", names_groups_file="names_groups.pkl")

        thesis.make_protein_groups()
        thesis.check_protein_groups()
        thesis.remove_interior_nodes()

        thesis.propagate_features_graph(steps=3)
        thesis.make_diffused_features() #definitely not
        thesis.make_targets() # we can probably use this
        if filter_node_amount is not None:
            thesis.filter_by_node_amount(amount=filter_node_amount)

        thesis.run_cv_hypersearch_xgb(3)

    def __init__(self, filename):
        self.experiment_name = filename.stem

        with open(filename) as file:
            self.config = yaml.safe_load(file)

        logging.info("Loading experiment with config %s", self.config)

        if self.config["model"] == "gcn":
            self.model = self.run_gcn
        elif self.config["model"] == "xgb":
            self.model = self.run_xgb
        else:
            raise ValueError(f"Model not understood: f{self.config['model']}")

        del self.config["model"]

    def run(self):
        self.model(**self.config)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="experiment_executor.log")
    logger = logging.getLogger()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for filename in pathlib.Path("experiments").glob("*.yml"):

        if "xgb" in str(filename) or filename.stem in ["simple_gcn", "normalize_gcn", "neighborhood_gcn"]:
            print(f"Skipping {filename}")
            continue
        # Restart logger with a new log file
        logger.removeHandler(logger.handlers[-1])
        handler = logging.FileHandler(f"logs/{filename.stem}.log")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Run experiment
        logging.info(f"Running experiment {filename.stem}")
        experiment = ExperimentExecutor(filename)
        experiment.run()
        logging.info(f"Finished experiment {filename.stem}")

