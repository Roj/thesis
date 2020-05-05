import yaml
import pathlib
from thesispipeline import ThesisPipeline

class ExperimentExecutor:
    def run_gcn(self, experiment_name, contacts=False, neighborhoods=False,
                normalize_adj=False, epochs=2, filter_node_amount=None):
        thesis = ThesisPipeline()

        if contacts:
            thesis.load_data(graphs_folder="graphs/contacts/", names_groups_file="names_groups.pkl")
            thesis.filter_graphs_missing_data()
        else:
            thesis.load_data(graphs_folder="graphs/", names_groups_file="names_groups.pkl")

        thesis.make_protein_groups()
        thesis.check_protein_groups()
        thesis.remove_interior_nodes()
        thesis.make_features()
        thesis.make_adjacency_matrices()
        thesis.make_targets()
        thesis.convert_matrices_32bits()

        if filter_node_amount is not None:
            thesis.filter_by_node_amount(amount=filter_node_amount)

        if neighborhoods:
            thesis.make_neighborhoods(dist=3, negative_prob=0.5, verbose=1)

        thesis.make_positive_weight()
        thesis.make_laplacians()
        thesis.pad_matrices()
        if neighborhoods:
            thesis.run_cv_local_gcn(epochs=epochs)
        else:
            thesis.make_masks()
            thesis.run_cv_gcn(epochs=epochs)

    def run_xgb(self, experiment_name, contacts=False, steps=3, filter_node_amount=None):
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
        # TODO: log config
        with open(filename) as file:
            self.config = yaml.safe_load(file)

        if self.config["model"] == "gcn":
            self.model = self.run_gcn
        elif self.config["model"] == "xgb":
            self.model = self.run_xgb
        else:
            raise ValueError(f"Model not understood: f{self.config['model']}")

        del self.config["model"]

    def run(self):
        self.model(self.experiment_name, **self.config)

if __name__ == "__main__":
    # change log according to experiment
    for filename in pathlib.Path("experiments").glob("*.yml"):
        experiment = ExperimentExecutor(filename)
        experiment.run()