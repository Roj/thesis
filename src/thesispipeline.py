"""Main pipeline options for experiments"""
import os
import logging
import subprocess
import progressbar
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd
import tensorflow as tf
from gcn.gcn import Laplacian, sparse_to_tuple, GraphConvolutionalNetwork, LocalGCN
from gcn.hyperparameterdetective import HyperparameterDetective

class ThesisPipeline:
    def __init__(self):
        return None

    def load_data(self, graphs_folder="../graphs", names_groups_file = "../names_groups.pkl"):
        """Load graphs and names/groups from CDHit"""
        import pickle
        def load_graph(fn):
            with open(fn, "rb") as f:
                return pickle.load(f)
        self.filenames = [fn for fn in os.listdir(graphs_folder) if ".pkl" in fn]
        self.graphs = [load_graph(os.path.join(graphs_folder, fn)) for fn in self.filenames]
        with open(names_groups_file, "rb")  as f:
            self.names, self.groups = pickle.load(f)
    def make_protein_groups(self):
        """Coalesce chain-level groups into protein-level groups"""
        from biograph.groupfolds import CDHitGroup
        protein_group_dict = CDHitGroup.get_protein_groups(self.names, self.groups)
        self.protein_ids = []
        for graph in self.graphs:
            node_idx = list(graph.nodes)[0]
            protein_id = graph.nodes[node_idx]["chain"].split("_")[0]
            self.protein_ids.append(protein_id)
        self.protein_groups = [protein_group_dict.get(pdbid, -1) for pdbid in self.protein_ids]
        logging.info(f"There are {len(list(filter(lambda g: g==-1, self.protein_groups)))}"
                    "proteins without group")

    def check_protein_groups(self):
        """Adds group information to graph nodes and checks how many nodes do not have
        a group (i.e. CDHit ignored it because of an error or otherwise)"""
        notin = 0
        total = 0
        chainnot = {}
        for graph in self.graphs:
            for node_idx in graph.nodes:
                total+=1
                chain = graph.nodes[node_idx]["chain"]
                if chain not in self.groups:
                    notin+=1
                    graph.nodes[node_idx]["group"] = None
                    chainnot[chain] = chainnot.get(chain, 0) + 1
                else:
                    graph.nodes[node_idx]["group"] = chain
        logging.info(f"Nodes w/o CDHit group: {notin} ({notin/total *100}%)\nBy chain:{chainnot}")
    def remove_interior_nodes(self):
        """Remove interior nodes from topological graphs"""
        def remove_interior(graph):
            edges_to_remove = set()
            nodes_to_remove = set()
            for node_idx, adj_dict in graph.adjacency():
                neighbors_not_in_surf = [k for k,v in adj_dict.items() if not v["in_surf"]]
                edges_to_remove.update([
                    (node_idx, neighbor)  if node_idx < neighbor else (neighbor, node_idx)
                    for neighbor in neighbors_not_in_surf])
                if len(adj_dict) == len(neighbors_not_in_surf):
                    nodes_to_remove.add(node_idx)
            #edges_before = graph.number_of_edges()
            #nodes_before = graph.number_of_nodes()
            for edge in edges_to_remove:
                graph.remove_edge(*edge)
            for idx in nodes_to_remove:
                graph.remove_node(idx)
            return graph
        self.graphs = [remove_interior(graph) for graph in self.graphs]
    def filter_graphs_missing_data(self):
        """Keep only graphs that do not have nodes with missing data.
        Missing data can be checked by asserting if 'chain' is in the node dict."""
        has_missing_data = []
        for i, graph in enumerate(self.graphs):
            nodes_without_data = False
            for node_idx in graph.nodes:
                if "chain" not in graph.nodes[node_idx]:
                    nodes_without_data = True
                    break
            has_missing_data.append(nodes_without_data)
        graphs_missing_data = len([_ for _ in has_missing_data if _])
        logging.info(f"Number of graphs missing data: {graphs_missing_data} "
                     f"({graphs_missing_data/len(self.graphs)*100}%)\n"
                     f"Remaining graphs: {len(self.graphs)-graphs_missing_data}")
        self.graphs = [graph for i, graph in enumerate(self.graphs) if not has_missing_data[i]]
    def make_features(self):
        """Make features from graph data
            - one shot for resname (including UNK!)
            - bfactor and coord as usual"""
        import biograph.constants
        index_amino = {code3:i for i, code3 in enumerate(biograph.constants.AMINOACIDS_3)}
        index_amino["UNK"] = len(index_amino)
        num_amino = len(index_amino)

        # Features are aminoacid type, bfactor and x,y,z coord.
        self.all_features = []
        for graph in self.graphs:
            features = np.zeros((graph.number_of_nodes(), num_amino + 4))
            for i, node_idx in enumerate(graph.nodes):
                node = graph.nodes[node_idx]
                #features[i, 0:num_amino+4] = 1
                features[i, index_amino[node["resname"]]] = 1
                features[i, num_amino] = node["bfactor"]
                if "coord" in node:
                    features[i, num_amino+1:num_amino+4] = node["coord"]
                else:
                    features[i, num_amino+1:num_amino+4] = node["x"], node["y"], node["z"]

            self.all_features.append(features)

    def propagate_features_graph(self, steps=3):
        """Propagate feature values along edges"""
        from biograph.graph_models import GraphModel
        logging.info("Propagating features...")
        for i in progressbar.progressbar(range(len(self.graphs))):
            self.graphs[i] = GraphModel.get_diffused_graph(
                self.graphs[i], steps=steps)

    def make_diffused_features(self, steps=3):
        """Obtain diffused features from already diffused graphs"""
        from biograph.graph_models import GraphModel
        import biograph.constants

        self.all_features = []
        logging.info("Generating features..")
        for graph in progressbar.progressbar(self.graphs):
            df = GraphModel.graph_to_dataframe(graph)
            # One hot for AAs
            for code3 in biograph.constants.AMINOACIDS_3 + ["UNK"]:
                df[code3] = (df.resname == code3).astype(np.int)

            if "coord" in df.columns:
                for i, coord in enumerate(["x", "y", "z"]):
                    df[coord] = df.coord.map(lambda x: x[i])

            df = df[[c for c in df.columns if "distance_" not in c]]
            df = df.drop(["full_id","resname", "coord", "distance", "chain"],
                         axis=1, errors="ignore")
            if not self.all_features:
                logging.info(f"Dataframe features: {df.columns}")
            self.all_features.append(df)


    def plot_features_shape_hist(self):
        """Plot number of rows for features (i.e. node amount)"""
        import matplotlib.pyplot as plt
        plt.hist([features.shape[0] for features in self.all_features])

    def make_adjacency_matrices(self):
        """Make adjacency matrix using a sparse representation"""
        self.all_adj = [nx.adjacency_matrix(graph) for graph in self.graphs]

    def normalize_adjacency_matrices(self):
        """Normalize adjacency matrices into 1/0 assuming CSR format"""
        for A in self.all_adj:
            A.data = np.ones_like(A.data, dtype=A.data.dtype)

    def make_targets(self):
        """Make targets using customary definition based on distance to ligand"""
        def touches_ligand(x):
            return x <= 4 or (x<=6 and np.random.binomial(1, 1-(x-4)/2) == 1)

        self.class_balance = []
        self.all_targets = []
        for graph in self.graphs:
            targets = np.zeros((graph.number_of_nodes()))
            for i, node_idx in enumerate(graph.nodes):
                distance = graph.nodes[node_idx]["distance"]
                targets[i] = 1 if touches_ligand(distance) else 0
            self.class_balance.append(targets.sum() / (targets.shape[0]- targets.sum()))
            self.all_targets.append(targets)

    def filter_by_node_amount(self, amount=1000):
        """Discard instances that have more than `amount` nodes. Discards
        features, adjacency matrices, targets and protein groups."""
        logging.info(f"Before filtering: {len(self.all_features)} instances")
        keep = [features.shape[0] < amount for features in self.all_features]
        self.all_features = [features
            for i, features in enumerate(self.all_features)
            if keep[i]]

        self.all_targets = [targets
                            for i, targets in enumerate(self.all_targets)
                            if keep[i]]
        self.protein_groups = [g for i, g in enumerate(self.protein_groups)
                              if keep[i]]
        if "all_adj" in self.__dict__:
            self.all_adj = [adj
                            for i, adj in enumerate(self.all_adj)
                            if keep[i]]

        logging.info(f"After filtering: {len(self.all_features)} instances")

    def convert_matrices_32bits(self):
        """Convert matrices to 32bits to not burn RAM"""
        self.all_features = [features.astype(np.float32)
                             for features in self.all_features]

        self.all_targets = [targets.astype(np.float32)
                            for targets in self.all_targets]
        if "all_adj" in self.__dict__:
            self.all_adj = [adj.astype(np.float32)
                            for adj in self.all_adj]

    def make_neighborhoods(self, nb_nodes=None, dist=2, negative_prob=0.5, verbose=0):
        """Explode features, targets, adjacencies and masks into neighborhood-level.
        Also adjusts class balance and protein groups.

        Parameters
        ----------
            nb_nodes: int
                Specify to filter neighborhoods bigger than the
                given size and pad arrays to that amount.
            dist: int
                Number of steps to take from center node to create
                a neighborhood.
            negative_prob: float
                Sampling probability for a negative node.
            verbose: int
                Higher number is more verbose."""
        from neighborhooditerator import NeighborhoodIterator
        neighborhood_features = []
        neighborhood_adj = []
        neighborhood_targets = []
        neighborhood_masks = []
        neighborhood_groups = []
        neighborhood_last_indicator = []
        class_balance = [0, 0]

        if "all_adj" in self.__dict__:
            del self.all_adj


        logging.info("Generating neighborhoods..")
        it = self.graphs
        if verbose > 0:
            it = progressbar.ProgressBar()(it, max_value=len(self.graphs))
        for graph in it:
            graph_features = self.all_features.pop(0)
            graph_targets = self.all_targets.pop(0)
            graph_group = self.protein_groups.pop(0)
            num_neighborhoods = 0
            for nA, nF, nT, nM in NeighborhoodIterator(
                    graph, graph_features, graph_targets, nb_nodes=nb_nodes,
                    dist=dist, negative_prob=negative_prob):

                # since mask only selects the center node, sum(mask*targets)
                # gets that single value
                class_balance[np.isclose((nM*nT).sum(),1)] += 1

                neighborhood_adj.append(nA)
                neighborhood_features.append(nF)
                neighborhood_targets.append(nT)
                neighborhood_masks.append(nM)
                neighborhood_groups.append(graph_group)
                num_neighborhoods+=1

            neighborhood_last_indicator.extend([0]*(num_neighborhoods-1) + [1])

        self.all_features = neighborhood_features
        self.all_targets = neighborhood_targets
        self.all_adj = neighborhood_adj
        self.all_masks = neighborhood_masks
        self.protein_groups = neighborhood_groups
        self.last_neighborhood = neighborhood_last_indicator

        self.class_balance = [class_balance[1]/class_balance[0]] # positive / negative
        logging.info("New class balance is %f", self.class_balance[0])


    def make_positive_weight(self):
        """Make positive class weight"""
        self.fair_positive_weight = 1/(sum(self.class_balance)/len(self.class_balance))
        logging.info(f"For every non-contact point there are "
                     f"{1/self.fair_positive_weight} contact points "
                     f"(positive class weight = {self.fair_positive_weight})")

    def make_laplacians(self):
        self.all_laplacians = [sparse_to_tuple(Laplacian.from_adjacency(adj))
            for adj in self.all_adj]

    def pad_matrices(self):
        """Make the matrices the same size via padding or modifying the sparse
        matrix attributes. """
        self.nb_nodes_per_graph = [adj[2][1] for adj in self.all_laplacians]
        self.nb_nodes = max(map(lambda adj: adj[2][1], self.all_laplacians))
        logging.info(f"Maximum number of nodes per graph: {self.nb_nodes}")
        # Make sparse matrices the same size
        for i, adj_tuple in enumerate(self.all_laplacians):
            #adj_tuple[2] is the shape, and we want it to be always the same..
            self.all_laplacians[i] = (adj_tuple[0], adj_tuple[1], (self.nb_nodes, self.nb_nodes))

        # Make features the same size as well via padding
        for i, features in enumerate(self.all_features):
            amount = self.nb_nodes-features.shape[0]
            self.all_features[i] = np.pad(features, ((0,amount), (0,0)))

        for i, target in enumerate(self.all_targets):
            amount = self.nb_nodes-target.shape[0]
            self.all_targets[i] = np.pad(target, (0,amount))

    def make_masks(self, pad_only=False):
        """Make masks. If using predefined masks, you can just pad them specifying pad_only."""
        if not pad_only:
            self.all_masks = []
            for n in self.nb_nodes_per_graph:
                mask = np.ones(n)
                self.all_masks.append(mask)

        for i, (mask, n) in enumerate(zip(self.all_masks, self.nb_nodes_per_graph)):
            self.all_masks[i] = np.pad(mask, (0, self.nb_nodes - n)).astype(np.float32)

    def merge_neighborhoods(self, max_nodes = 1000):
        """Since neighborhoods are usually small in size, we can merge them into
        fewer matrices speeding up computation. Laplacians are concatenated to be
        block-diagonal, while features, targets and masks are concatenated accross
        rows. Additionally, padding is done in this same method for simplicity.
        All matrices end up having `max_nodes` rows."""
        laplacians_concatenated = []
        features_concatenated = []
        targets_concatenated = []
        masks_concatenated = []
        groups_concatenated = []
        max_nodes = 1000
        num_processed = 0

        while num_processed < len(self.all_laplacians):
            indices, values, shape = self.all_laplacians[num_processed]
            features = [self.all_features[num_processed]]
            targets = [self.all_targets[num_processed]]
            masks = [self.all_masks[num_processed]]
            group = thesis.protein_groups[num_processed]

            indices = [indices]
            values = [values]
            nodes = shape[0]

            num_processed += 1
            while (num_processed < len(self.all_laplacians)) \
                and (group == self.protein_groups[num_processed]) \
                and (nodes + self.all_laplacians[num_processed][2][0] < max_nodes):
                idx, vals, shape = self.all_laplacians[num_processed]

                indices.append(idx + nodes)
                values.append(vals)
                features.append(self.all_features[num_processed])
                targets.append(self.all_targets[num_processed])
                masks.append(self.all_masks[num_processed])

                nodes += shape[0]
                num_processed += 1

            indices = np.concatenate(indices)
            values = np.concatenate(values)
            shape = (max_nodes, max_nodes) # "Padding" for sparse matrices

            padding_amount = max_nodes-nodes

            features = np.pad(np.concatenate(features), ((0, padding_amount), (0, 0)))
            targets = np.pad(np.concatenate(targets), (0, padding_amount))
            masks = np.pad(np.concatenate(masks), (0, padding_amount))

            laplacians_concatenated.append((indices, values, shape))
            features_concatenated.append(features)
            targets_concatenated.append(targets)
            masks_concatenated.append(masks)
            groups_concatenated.append(group)

        self.all_laplacians = laplacians_concatenated
        self.all_features = features_concatenated
        self.all_targets = targets_concatenated
        self.all_masks = masks_concatenated
        self.protein_groups = groups_concatenated

    def _prepare_tensors(self):
        feats = self.all_features
        supps = [tf.sparse.SparseTensor(indices, values.astype(np.float32), dense_shape)
                    for indices, values, dense_shape in self.all_laplacians]
        targs = self.all_targets
        masks = self.all_masks
        return feats, targs, supps, masks

    def run_cv_gcn(self, epochs=40, name=""):
        """Prepare tensors and run GCN with Group K-Fold CV"""
        feats, targs, supps, masks = self._prepare_tensors()

        model = GraphConvolutionalNetwork(feats[0].shape, 1, self.all_laplacians[0][2], name=name)
        model.fit_cv_groups((feats, targs, supps, masks), self.protein_groups,
                   positive_weight=self.fair_positive_weight, epochs=epochs)
        return model

    def run_hypersearch_gcn(self, hyperparameter_domains, prefix, epochs=40, name=""):
        """Run hyperparameter detective with given parameter domains. Prefix must be
        specified."""
        feats, targs, supps, masks = self._prepare_tensors()
        detective = HyperparameterDetective(
            f"logs/{prefix}", name, hyperparameter_domains,
            feats[0].shape, 1, self.all_laplacians[0][2],
            name=name
        )
        detective.search((feats, targs, supps, masks), self.protein_groups,
                         positive_weight=self.fair_positive_weight, epochs=epochs)

    def run_cv_local_gcn(self, epochs=40, name=""):
        """Prepare tensors and run LocalGCN with Group K-Fold CV"""
        feats, targs, supps, masks = self._prepare_tensors()
        last_neighborhood = self.last_neighborhood

        model = LocalGCN(feats[0].shape, 1, self.all_laplacians[0][2], name=name)
        model.fit_cv_groups((feats, targs, supps, masks, last_neighborhood), self.protein_groups,
                positive_weight=self.fair_positive_weight, epochs=epochs)
        return model

    def run_cv_hypersearch_xgb(self, n_iter):
        """Concatenate all dataframes and run XGB with Group K-Fold CV
        and Randomized Hyperparameter Search."""
        from scipy.stats import uniform
        import sklearn.model_selection
        import xgboost as xgb

        for df, group, target in zip(self.all_features, self.protein_groups, self.all_targets):
            df["target"] = target
            df["group"] = group
        dataset = pd.concat(self.all_features, ignore_index=True)
        dataset = dataset.loc[dataset.group.notna()]

        row_groups = dataset.group
        row_target = dataset.target
        dataset = dataset.drop(["group", "target"], axis=1)

        logging.info(f"Dataframe shape: {dataset.shape}")
        logging.info(f"Columns passed to model: {dataset.columns}")
        logging.info(f"Data types: {dataset.dtypes}")

        param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic',
                 'nthread': 4, 'eval_metric': 'auc'}

        groupk = sklearn.model_selection.GroupKFold(n_splits=5)
        clf = xgb.XGBClassifier(**param)

        clf = sklearn.model_selection.RandomizedSearchCV(
            clf, {"max_depth": list(range(3,10)),
                "min_child_weight": uniform(loc=0.5, scale=1.5), # ~U(loc, loc+scale)
                "eta": uniform(loc=0.2, scale=0.3)},
            cv=groupk, n_iter=n_iter, scoring="roc_auc", verbose=5, n_jobs=3)

        search = clf.fit(dataset, row_target, row_groups)

        return clf, search
