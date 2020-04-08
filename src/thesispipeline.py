import os
import logging
import networkx as nx
import numpy as np
import scipy.sparse as sp
import subprocess
import tensorflow as tf
from gcn.gcn import Laplacian, sparse_to_tuple, GraphConvolutionalNetwork

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

    def plot_features_shape_hist(self):
        """Plot number of rows for features (i.e. node amount)"""
        import matplotlib.pyplot as plt
        plt.hist([features.shape[0] for features in self.all_features])

    def make_adjacency_matrices(self):
        """Make adjacency matrix using a sparse representation"""
        # TODO: normalize?
        self.all_adj = [nx.adjacency_matrix(graph) for graph in self.graphs]

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
        keep = [features.shape[0] < amount for features in self.all_features]
        self.all_features = [features.astype(np.float32)
            for i, features in enumerate(self.all_features)
            if keep[i]]
        self.all_adj = [adj.astype(np.float32)
                        for i, adj in enumerate(self.all_adj)
                        if keep[i]]
        self.all_targets = [targets.astype(np.float32)
                            for i, targets in enumerate(self.all_targets)
                            if keep[i]]
        self.protein_groups = [g for i, g in enumerate(self.protein_groups)
                               if keep[i]]
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
        matrix attributes. Make masks as well"""
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

        self.masks_all = []
        for n in self.nb_nodes_per_graph:
            mask = np.pad(np.ones(n), (0, self.nb_nodes - n)).astype(np.float32)
            self.masks_all.append(mask)

    def run_cv_gcn(self, epochs=40):
        tf.data.Dataset.from_tensor_slices = lambda a: a
        feats = self.all_features
        supps = [tf.sparse.SparseTensor(indices, values.astype(np.float32), dense_shape)
                    for indices, values, dense_shape in self.all_laplacians]
        targs = self.all_targets
        masks = self.masks_all
        model = GraphConvolutionalNetwork(feats[0].shape, 1, self.all_laplacians[0][2])
        model.fit_cv_groups((feats, supps, targs, masks), self.protein_groups,
                   positive_weight=self.fair_positive_weight, epochs=epochs)