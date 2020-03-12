#!/usr/bin/python3

import os
import sys
import argparse
import requests
import multiprocessing
import warnings
import numpy as np
import networkx as nx
import pickle as pkl
import progressbar
import logging
from utils import suppress_stdout_stderr

with suppress_stdout_stderr():
    import vmd #suppress annoying warnings from plugins

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
parser.add_argument('--contacts_graph', dest='contacts_graph', action='store_true',
                    help='use a contacts graph instead of a surface graph')
parser.add_argument("--workers", dest="workers", type=int, default=4,
                    help='number of workers to use')
parser.add_argument('--skip_graph_generation', dest='skip_graph_generation', action="store_true",
                    help="skip graph generation altogether")
parser.add_argument("--discard-distant-chains", dest="discard_chains_without_atp", action="store_true",
                    help="discard chains that have no contact with ATP")



#logging.getLogger().addHandler(logging.StreamHandler())

args = parser.parse_args()


# Download datasets.
if not os.path.exists("data/consurf"):
    os.makedirs("data/consurf")

if not os.path.exists("graphs/contacts"):
    os.makedirs("graphs/contacts")

if not os.path.exists("if/contacts/"):
    os.makedirs("if/contacts")

if not os.path.exists("if/multiprocessing/"):
    os.makedirs("if/multiprocessing")


# Searching a bit I didn't find a way to get a chain list or get all
# conservation features for a specific protein, so we need to look up
# what chains it actually has by reading the PDB file.


def worker(filenames, progress_queue, logger, worker_id):
    # Set up logging
    logger.set_master(False)
    logger.set_pid(worker_id)
    logger.info("Started! Hello world.")
    sequences = []
    names = []
    names_mapping = {}
    #setup logging with worker_id
    for i, fn in enumerate(filenames):
        if fn is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                protein = Protein(fn)
            except Exception as e:
                logger.error(f"Error at protein #{i}/{fn}: {e}")
                continue

        # Add sequence and name to list
        if len(protein.sequences) == 0:
            logger.error(f"PDB #{i}/{protein.pdb.id} has no sequences!")
            continue

        sequences.extend(protein.sequences.values())
        names.extend(protein.sequences.keys())
        names_mapping[protein.pdb.id] = list(protein.sequences.keys())

        # Consurf
        # get_chains might have repeated IDs, so we use a set generator.
        chains = list({chain.id for chain in protein.pdb.get_chains()})
        id_chain_dict = {protein.pdb.id : chains}

        if not args.no_consurf:
            dw = ConsurfDBDownloader(id_chain_dict, base_path="data/consurf/")
            logger.debug(f"Downloading conservation data for Protein {protein.pdb.id}")
            try:
                consurf_filenames = dw.request_and_write()
                # Add the features to the protein object.
                features = {}
                for consurf_fn, chain in zip(consurf_filenames, chains):
                    if consurf_fn is not None:
                        features.update(protein.get_conservation_features(consurf_fn, [chain]))
                protein.add_residue_features(features)
            except requests.exceptions.ReadTimeout:
                logger.error(f"Timeout while fetching conservation data for protein {protein.pdb.id}")
                pass

        if args.skip_graph_generation:
            continue

        # Features and graph generation
        # Distance features
        protein.df = protein.df[~protein.df.coord.isnull()]
        ATP_coords = protein.df[protein.df.resname == "ATP"].coord.to_list()

        if len(ATP_coords) > 200 and args.laptop_safe:
            logger.info(f"{protein.pdb.id} - Skipping it because it would fry my laptop")
            continue
        if len(ATP_coords) == 0:
            # This may happen because of an oddly formatted PDB file which Bio cannot read
            # correctly.
            logger.warning(f"{protein.pdb.id} no ATP atoms found. Skipping.")
            continue
        protein.df["distance"] = protein.df.coord.apply(
            lambda atom: min(map(lambda atp_atom: np.linalg.norm(atom-atp_atom), ATP_coords))
        )
        protein.discard_ligands()
        # Sanity check
        protein.df = protein.df.loc[
                protein.df.apply(lambda row: row["full_id"][4][0] == "CA", axis=1),:].reset_index(drop=True)

        if min(protein.df.distance) >= 6.0:
            logger.warning(f"{protein.pdb.id} no atoms are linked to ligand")
            continue

        if args.discard_chains_without_atp:
            # Only keep chains that are connected to the ligand.
            chains_with_ligand = protein.df[protein.df.distance <= 6.0].chain.unique()
            protein.select_chains(chains_with_ligand)

        # Identify chain uniquely (useful later to match with CDHit)
        protein.df["chain"] = protein.pdb.id + "_" + protein.df.chain

        if args.contacts_graph:
            scgg = graph_models.StaticContactGraphGenerator()
            try:
                protein.generate_graph(scgg, {
                    "save_contact_filename": f"if/contacts/{protein.pdb.id}_contacts.tsv"
                })
            except Exception as e:
                logger.info(f"Error while generating contacts graph on protein #{i} ({fn}): {e}")
                continue

            scgg.add_features(protein.df, columns = [
                "x", "y", "z", "bfactor", "chain", "occupancy", "distance"
            ])
            graph = protein.graph
            dest_pickle = "graphs/contacts/{}.pkl".format(protein.pdb.id)
        else:
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

            # Rest of features
            structure_model.add_features(protein.df, columns = [
                "bfactor", "score", "color",
                "color_confidence_interval_high", "color_confidence_interval_low",
                "score_confidence_interval_high", "score_confidence_interval_low",
                "resname", "coord", "distance",
                "chain"
            ])

            dest_pickle = "graphs/{}.pkl".format(protein.pdb.id)
            graph = structure_model.G

        nx.write_gpickle(graph, dest_pickle)
        #logging.debug("Exported graph to {}".format(dest_pickle))
        progress_queue.put(worker_id)

    with open(f"if/multiprocessing/generate_dataset_{worker_id}", "wb") as intermediate_file:
        pkl.dump((sequences, names, names_mapping), intermediate_file)


def download_pdbs_get_filenames():
    """Returns the filenames of the PDB files, optionally
    fetching them from the database if args.skip_pdb_download
    is false"""
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

    return filenames
    #filenames = ["data/3KEU.pdb", "data/3ULE.pdb", "dlata/4DQW.pdb"]

def run_cdhit(sequences, names, names_mapping):
    groups = CDHitGroup.get_group_by_sequences(sequences, names)
    groups_mapping = {name:groups[i] for i, name in enumerate(names)}

    with open("names_groups.pkl", "wb") as f:
        pkl.dump((names_mapping, groups_mapping), f)

class LoggerProcess():
    def __init__(self, level = logging.DEBUG):
        logging.basicConfig(level=logging.DEBUG, filename="generate_dataset.log")
        self.queue = multiprocessing.Queue()
        self.pid = None
        self.master = False
        self.stop_token = (-1, "STOP")

    def listen(self):
        """This should only be called as target of a new process"""
        #logging.getLogger().addHandler(logging.StreamHandler())
        logging.debug("Test2 logger")
        self._log(logging.DEBUG, "[LOGGER] started")
        while True:
            msg_level, msg = self.queue.get()
            if self.stop_token == (msg_level, msg):
                self._log(logging.DEBUG, "[LOGGER] Received stop, stopping")
                break
            self._log(msg_level, msg)

    def request_stop(self):
        self.queue.put(self.stop_token)

    def set_pid(self, pid):
        self.pid = pid

    def set_master(self, master):
        self.master = master

    def _log(self, level, msg):
        method = logging.debug
        if level == logging.INFO:
            method = logging.info
        elif level == logging.WARNING:
            method = logging.warning
        elif level == logging.ERROR:
            method = logging.error

        method(msg)

    def _send_log(self, level, msg):
        prefix = "[MASTER]"
        if not self.master:
            prefix = f"[WORKER {self.pid}]"
        self.queue.put((level, f"{prefix} {msg}"))

    def error(self, msg):
        self._send_log(logging.ERROR, msg)

    def warning(self, msg):
        self._send_log(logging.WARNING, msg)

    def info(self, msg):
        self._send_log(logging.INFO, msg)

    def debug(self, msg):
        self._send_log(logging.DEBUG, msg)


def master():
    # Set up logging
    logger = LoggerProcess()
    logger_process = multiprocessing.Process(target=logger.listen)
    logger_process.start()
    logger.set_master(True)

    logger.debug("Downloading files")
    filenames = download_pdbs_get_filenames()
    progress_queue = multiprocessing.Queue()

    logger.debug("Creating workers")
    children = []
    batch_size = int(len(filenames)/args.workers)
    jobs_quota = [] # track how many proteins each process has to.. process
    for j in range(args.workers):
        filenames_segment = filenames[j*batch_size:(j+1)*batch_size]
        if j == args.workers-1:
            filenames_segment = filenames[j*batch_size:]
        jobs_quota.append(len(filenames_segment))
        children.append(multiprocessing.Process(
            target=worker,
            args=(filenames_segment, progress_queue, logger, j)))


    logger.debug("Starting workers")
    for child in children:
        child.start()

    bar = progressbar.ProgressBar(max_value=len(filenames))
    i = 0
    jobs_processed = [0]*len(jobs_quota)
    while bar.value < len(filenames):
        pnum = progress_queue.get()
        jobs_processed[pnum] += 1
        bar.update(bar.value + 1)
        for j, child in enumerate(children):
            if not child.is_alive() and jobs_processed[j] < jobs_quota[j]:
                logger.error(f"Process {j} is not alive and"
                    f"processed {jobs_processed[j]} out of {jobs_quota[j]}")
                raise Exception # restart process?

    logger.info("All proteins processed, waiting for children to finish")
    for i, child in enumerate(children):
        logger.debug(f"Waiting for child #{i}")
        child.join()

    logger.info("Children finished, joining results")
    sequences = []
    names = []
    names_mapping = {}
    for j in range(args.workers):
        worker_seq, worker_names, worker_mapping = pkl.load(f"if/multiprocessing/generate_dataset_{j}")
        sequences.extend(worker_seq)
        names.extend(worker_names)
        names_mapping.update(worker_mapping)

    logger.info("Running CDHit")

    run_cdhit(sequences, names, names_mapping)


    logger.info("CDHit finished")

    logger.debug("Asking logger to stop")
    logger.request_stop()
    logger_process.join()

if __name__ == "__main__":
    master()
