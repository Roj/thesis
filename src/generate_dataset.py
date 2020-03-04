#!/usr/bin/python3

import os
import sys
import argparse
import requests
import warnings
import numpy as np
import networkx as nx
import pickle as pkl
import progressbar

import biograph.graph_models as graph_models
from biograph.downloader import PdbDownloader, ConsurfDBDownloader
from biograph.protein import Protein
from biograph.structure import Perseus
from biograph.groupfolds import CDHitGroup

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate dataset for protein graph analysis.')
parser.add_argument('--skip-pdb-download', dest='skip_pdb_download', action='store_true',
                    help='skip PDB download')
parser.add_argument('--no-consurf', dest='no_consurf', action='store_true',
                    help='do not add conservation (consurf DB) data')
parser.add_argument('--laptop-safe', dest='laptop_safe', action='store_true',
                    help='do not process proteins with > 200 ATP connections')
parser.add_argument('--include-interior-points', dest='include_interior', action='store_true',
                    help='do not process proteins with > 200 ATP connections')



args = parser.parse_args()

# Download datasets.
if not os.path.exists("data/consurf"):
    os.makedirs("data/consurf")

if not os.path.exists("graphs/"):
    os.makedirs("graphs/")

if not args.skip_pdb_download:
    with open("pdb_atp_list.txt", "r") as f:
        # IDs are in one line separated by commas
        pdb_list = [id.strip() for id in f.readline().split(",")]

    downloader = PdbDownloader(pdb_list, base_path="data/")
    filenames = downloader.request_and_write()
else:
    filenames = os.listdir("data/")
    filenames = ["data/"+fn for fn in filenames if ".pdb" in fn]

if any(map(lambda fn: fn is None, filenames)):
    print("Some files could not be downloaded")

#filenames = ["data/3KEU.pdb", "data/3ULE.pdb", "dlata/4DQW.pdb"]
# Searching a bit I didn't find a way to get a chain list or get all
# conservation features for a specific protein, so we need to look up
# what chains it actually has by reading the PDB file.
print("Loading proteins")
names = []
sequences = []
names_mapping = {}
for i, fn in progressbar.progressbar(enumerate(filenames), max_value=len(filenames)):

    if fn is None:
        continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        protein = Protein(fn)

    # get_chains might have repeated IDs, so we use a set generator.
    chains = list({chain.id for chain in protein.pdb.get_chains()})
    id_chain_dict = {protein.pdb.id : chains}

    if not args.no_consurf:
        dw = ConsurfDBDownloader(id_chain_dict, base_path="data/consurf/")
        print("Downloading conservation data for Protein {}".format(protein.pdb.id))
        try:
            consurf_filenames = dw.request_and_write()
            # Add the features to the protein object.
            features = {}
            for consurf_fn, chain in zip(consurf_filenames, chains):
                if consurf_fn is not None:
                    features.update(protein.get_conservation_features(consurf_fn, [chain]))
            protein.add_residue_features(features)
        except requests.exceptions.ReadTimeout:
            print("Timeout while fetching conservation data")
            pass
    # Distance features
    protein.df = protein.df[~protein.df.coord.isnull()]
    ATP_coords = protein.df[protein.df.resname == "ATP"].coord.to_list()

    if len(ATP_coords) > 200 and args.laptop_safe:
        print("Skipping it because it would fry my laptop")
        continue
    if len(ATP_coords) == 0:
        # This may happen because of an oddly formatted PDB file which Bio cannot read
        # correctly.
        print("WARNING: no ATP atoms found. Skipping.")
        continue
    protein.df["distance"] = protein.df.coord.apply(
        lambda atom: min(map(lambda atp_atom: np.linalg.norm(atom-atp_atom), ATP_coords))
    )
    protein.discard_ligands()
    # Sanity check
    protein.df = protein.df.loc[
            protein.df.apply(lambda row: row["full_id"][4][0] == "CA", axis=1),:].reset_index(drop=True)

    if min(protein.df.distance) >= 6.0:
        print("WARNING: no atoms are linked to ligand")
        continue

    # Only keep chains that are connected to the ligand.
    chains_with_ligand = protein.df[protein.df.distance <= 6.0].chain.unique()
    protein.select_chains(chains_with_ligand)

    # Generate the graph.
    structure = protein.generate_structure(lambda row: row["full_id"][4][0] == "CA")

    perseus = Perseus()
    perseus.execute_persistent_hom(protein)

    structure_model = graph_models.StructureGraphGenerator()
    protein.generate_graph(structure_model,
        {"step": structure.persistent_hom_params["b3_step"],
         "surface_only": not args.include_interior})

    # Depth features
    depths, _ = structure.calculate_depth(protein.graph)
    for node_idx, depth in depths.items():
        protein.graph.nodes[node_idx]["depth"] = depth

    # Identify chain uniquely (useful later to match with CDHit)
    protein.df["chain"] = protein.pdb.id + "_" + protein.df.chain
    # Rest of features
    structure_model.add_features(protein.df, columns = [
        "bfactor", "score", "color",
        "color_confidence_interval_high", "color_confidence_interval_low",
        "score_confidence_interval_high", "score_confidence_interval_low",
        "resname", "coord", "distance",
        "chain"
    ])

    dest_pickle = "graphs/{}.pkl".format(protein.pdb.id)
    nx.write_gpickle(structure_model.G, dest_pickle)
    #print("Exported graph to {}".format(dest_pickle))

    # Add sequence and name to list
    if len(protein.sequences) == 0:
        print("PDB {} has no sequences!".format(protein.pdb.id))
        continue
    sequences.extend(protein.sequences.values())
    names.extend(protein.sequences.keys())
    names_mapping[protein.pdb.id] = list(protein.sequences.keys())


print("Calculating CDHit groups..",end="")
groups = CDHitGroup.get_group_by_sequences(sequences, names)
print("Done! Writing files")
groups_mapping = {name:groups[i] for i, name in enumerate(names)}

with open("names_groups.pkl", "wb") as f:
    pkl.dump((names_mapping, groups_mapping), f)

