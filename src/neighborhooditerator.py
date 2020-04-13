"""Provides a class to make neighborhood patches from a graph"""
import numpy as np
import networkx as nx

class NeighborhoodIterator:
    """Class that takes an nx.Graph and explodes it into different neighborhoods
    with an iterator interface. Also handles node features and targets."""
    def __init__(self, graph, features, targets, nb_nodes=None, dist=2, negative_prob=1.0):
        """Sets up the iterator for the graph to transform into patches.
        Supports negative sampling with probability.
        Parameters
        ----------
            graph: nx.Graph
            features: list-like, aligned with graph.nodes
            targets: list-like, aligned with graph.nodes
            nb_nodes: int, number of nodes. If subgraph has fewer, padding will be used.
            dist: int, maximum distance to center node to consider for neighborhood
            negative_prob: float between 0 and 1, probabilty to accept a negative sample.
        Returns
        -------
            NeighborhoodIterator object"""
        self.G = graph
        self.features = features
        self.targets = targets
        self.nb_nodes = nb_nodes
        self.dist = dist
        self.negative_prob = negative_prob
        self.current = 0
        self.nodelist = list(graph.nodes)
        self._make_features_map()

    def _make_features_map(self):
        """Makes a dictionary mapping node idx to features"""
        self.nodeidx_features_map = dict()
        for node_idx, Fi in zip(self.nodelist, self.features):
            self.nodeidx_features_map[node_idx] = Fi

    def _get_neighborhood_nodes(self, graph, origin, maxdist):
        """Get a `maxdist`-neighborhood node list starting from `origin` for given `graph`."""
        neighborhood_nodes = {origin}
        dist = {origin: 0}
        queue = [origin]
        # Breadth First Search
        while queue:
            vertex = queue.pop()
            for neighbor_vertex in graph.neighbors(vertex):
                if (neighbor_vertex not in neighborhood_nodes) and (dist[vertex] < maxdist):
                    # Only insert into queue if distance is not exceeded in reaching it
                    # We avoid re-inserting via the set()
                    neighborhood_nodes.add(neighbor_vertex)
                    queue.append(neighbor_vertex)
                dist[neighbor_vertex] = dist.get(neighbor_vertex, dist[vertex] + 1)

        return neighborhood_nodes

    def __iter__(self):
        return self

    def __next__(self):
        """Returns adjacency matrix, features and center target for the
        neighborhood.

        Parameters
        ----------
            None
        Returns
        -------
            A: Adjacency Matrix (SciPy sparse matrix)
            F: features, list or numpy.ndarray
            t: center target. Array of len(nb_nodes) if nb_nodes is not None. Otherwise,
               it is just the value of the center node.
            mask: mask indicating center node"""
        self.current += 1
        if self.current > len(self.nodelist):
            raise StopIteration

        # Negative sampling
        if (np.isclose(self.targets[self.current-1], 0.0)) and (
                np.random.rand() > self.negative_prob):
            return self.__next__()

        node_idx = self.nodelist[self.current-1]
        neighbors = self._get_neighborhood_nodes(self.G, node_idx, self.dist)

        H = self.G.subgraph(neighbors)

        # Features and mask
        F = []
        mask = []
        for idx in H.nodes:
            F.append(self.nodeidx_features_map[idx])
            mask.append(1 if idx == node_idx else 0) # Center node

        targets = self.targets[self.current-1]
        # If we received a numpy array we should give back numpy arrays.
        # Otherwise, return a list.
        if isinstance(self.features, np.ndarray):
            F = np.array(F)
            mask = np.array(mask)
            if self.nb_nodes is not None:
                # We should pad all that can be padded (features, mask, targets).
                pad_amount = self.nb_nodes - F.shape[0]
                F = np.pad(F, ((0,pad_amount), (0,0)))
                mask = np.pad(mask, ((0, pad_amount)))
                targets = np.array([targets]*self.nb_nodes)

        return nx.adjacency_matrix(H), F, targets, mask