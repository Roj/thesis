#!/usr/bin/python3

import os
import sys
import argparse
import requests
import numpy as np
import networkx as nx

sys.path.append('../pyprot/')
import pyprot.graph_models as graph_models
from pyprot.downloader import PdbDownloader, ConsurfDBDownloader
from pyprot.protein import Protein
from pyprot.structure import Perseus

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate dataset for protein graph analysis.')
parser.add_argument('--skip-pdb-download', dest='skip_pdb_download', action='store_true',
                    help='skip PDB download')
parser.add_argument('--no-consurf', dest='no_consurf', action='store_true',
                    help='do not add conservation (consurf DB) data')


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


# Searching a bit I didn't find a way to get a chain list or get all
# conservation features for a specific protein, so we need to look up
# what chains it actually has by reading the PDB file.
proteins = []
for fn in filenames:
    if fn is None:
        proteins.append(None)
        continue
    protein = Protein(fn)
    proteins.append(protein)
    # get_chains might have repeated IDs, so we use a set generator.
    chains = list({chain.id for chain in protein.pdb.get_chains()})
    id_chain_dict = {protein.pdb.id : chains}
    dw = ConsurfDBDownloader(id_chain_dict, base_path="data/consurf/")

    if not args.no_consurf:
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
    if len(ATP_coords) == 0:
        # This may happen because of an oddly formatted PDB file which Bio cannot read
        # correctly.
        print("WARNING: no ATP atoms found. Skipping.")
        continue
    protein.df["distance"] = protein.df.coord.apply(
        lambda atom: min(map(lambda atp_atom: np.linalg.norm(atom-atp_atom), ATP_coords))
    )
    # Sanity check
    if min(protein.df[protein.df.resname != "ATP"].distance) > 4.0:
        print("WARNING: no atoms are linked to ligand")
        continue

    # Only keep chains that are connected to the ligand.
    chains_with_ligand = protein.df[protein.df.distance <= 6.0].chain.unique()
    protein.select_chains(chains_with_ligand)

    # Generate the graph.
    protein.discard_ligands()
    protein.generate_structure(lambda row: row["full_id"][4][0] == "CA")

    perseus = Perseus()
    perseus.execute_persistent_hom(protein)

    structure_model = graph_models.StructureGraphGenerator()
    protein.generate_graph(structure_model, {"step":389})
    structure_model.add_features(protein.df, columns = [
        "bfactor", "score", "color",
        "color_confidence_interval_high", "color_confidence_interval_low",
        "score_confidence_interval_high", "score_confidence_interval_low",
        "resname", "coord", "distance"
    ])
    dest_pickle = "graphs/{}.pkl".format(protein.pdb.id)
    nx.write_gpickle(structure_model.G, dest_pickle)
    print("Exported graph to {}".format(dest_pickle))


