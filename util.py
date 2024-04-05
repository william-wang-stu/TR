import datetime
import logging
from typing import Any, Dict, List, Tuple, NewType
import numpy as np
import os
import igraph
import math
import pickle

from param import *

CASCADES_TYPE = NewType('cascades_type', Dict[str, List[Any]])

# logging config
def Beijing_TimeZone_Converter(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp
# logging.Formatter.converter = time.gmtime
logging.Formatter.converter = Beijing_TimeZone_Converter

# dict mapping node_id to igraph.graph id
def get_graphid(node, node2graphList: Dict[str,str], readonly = False):
    if node not in node2graphList:
        if readonly:
            return -1
        new_graphid = len(node2graphList)
        node2graphList[node] = new_graphid
    return node2graphList[node]

def add_diffusion(diffusion_dict: Dict[str, List], user, diffusion: str, timestamp):
    if diffusion not in diffusion_dict:
        diffusion_dict[diffusion] = []
    diffusion_dict[diffusion].append((user, timestamp))

def init_graph(node2graphid_map: Dict[str, Any], edgelist: List[Any], is_directed=False, save_graph=False):
    graph = igraph.Graph(len(node2graphid_map), directed=False)
    graph.add_edges(edgelist)
    if not is_directed:
        graph.to_undirected()
    graph.simplify()

    if save_graph:
        # Save Graph in IGraph Format
        os.makedirs(outputfile_dirpath, exist_ok=True)
        with open(os.path.join(outputfile_dirpath, "igraph_edgelist"), "w") as f:
            graph.write(f, format="edgelist")
        logger.info("Save Network in IGraph Format")

    # add some fake vertices 
    # in case there's not enough vertices for subgraph generation
    graph.add_vertices(ego_size)

    return graph

# Save & Read Network / ActionLog
def build_network_from_edgelist():
    graph_node2graphid_map = {}
    graph_edgelist = []

    with open(edgelist_filepath, "r") as f:
        for _, line in enumerate(f):
            # 文件中出现非数字开头的内容就跳过这一行
            if line[0] < '0' or line[0] > '9':
                continue
            
            # Fix: 分隔符不确定是空格还是逗号
            node_pair = line[:-1].split(' ')
            if len(node_pair) < 2:
                node_pair = line[:-1].split(',')
            
            graph_edgelist.append(
                (
                    get_graphid(node_pair[0], graph_node2graphid_map),
                    get_graphid(node_pair[1], graph_node2graphid_map)
                )
            )
    logger.info(f"Load Edgelist from File {edgelist_filepath}, Graph is Composed of {len(graph_node2graphid_map)} Nodes and {len(graph_edgelist)} Edges")
    
    return graph_node2graphid_map, graph_edgelist

def save_network(node2graphid_map: Dict[str, str], edgelist: List[Any]):
    os.makedirs(outputfile_dirpath, exist_ok=True)
    with open(os.path.join(outputfile_dirpath, "node2graphid_map.txt"), "w") as f:
        for key, value in node2graphid_map.items():
            f.write(f"{key} {value}\n")
    logger.info("Save node2graphid_map")

    with open(os.path.join(outputfile_dirpath, "edgelist.txt"), "w") as f:
        for u, v in edgelist:
            f.write(f"{u} {v}\n")
    logger.info("Save edgelist")

def read_network(is_directed=False):
    graph_node2graphid_map = {}
    graph_edgelist = []

    with open(os.path.join(outputfile_dirpath, "node2graphid_map.txt"), "r") as f:
        for line in f:
            key, value = line[:-1].split(' ')
            graph_node2graphid_map[key] = int(value)
    logger.info("Read node2graphid_map from file %s" % (os.path.join(outputfile_dirpath, "node2graphid_map.txt")))

    with open(os.path.join(outputfile_dirpath, "edgelist.txt"), "r") as f:
        for line in f:
            from_node, to_node = line[:-1].split(' ')
            graph_edgelist.append((int(from_node), int(to_node)))
    logger.info("Read edgelist from file %s" % (os.path.join(outputfile_dirpath, "edgelist.txt")))

    return init_graph(graph_node2graphid_map, graph_edgelist, is_directed=is_directed)

def build_actionlog(node2graphid_map: Dict[str, str]):
    diffusion_dict = {}
    if actionlog_filepath.split('.')[1] == 'p':
        with open(actionlog_filepath, "rb") as f:
            diffusion_dict = pickle.load(f)
    else:
        with open(actionlog_filepath, "r") as f:
            for line in f:
                user_id, hashtag, timestamp = line[:-1].split(' ')
                graph_id = get_graphid(user_id, node2graphid_map, readonly=True)
                if graph_id > -1:
                    add_diffusion(diffusion_dict, graph_id, hashtag, timestamp)
    logger.info(f"Load ActionLog from File {actionlog_filepath}, ActionLog is Composed of {len(diffusion_dict)} Diffusions")

    # Sort by Timestamp
    for diffusion in diffusion_dict:
        diffusion_dict[diffusion] = sorted(diffusion_dict[diffusion], key = lambda item: int(item[1]))
    logger.info("Sort ActionLog by Timestamp")

    return diffusion_dict

def save_actionlog(diffusion_dict: Dict[str, Any]):
    # Format: hashtag user1,timestamp1 user2,timestamp2 ...
    os.makedirs(outputfile_dirpath, exist_ok=True)
    with open(os.path.join(outputfile_dirpath, "sorted_actionlog"), "w") as f:
        for key, value in diffusion_dict.items():
            f.write(f"{key}")
            for user, timestamp in value:
                f.write(f" {user},{timestamp}")
            f.write("\n")
    logger.info(f"Save sorted_actionlog with {len(diffusion_dict)} Diffusions")

def save_actionlog2(diffusion_dict: Dict[str, Any], graph_vcount: int):
    ### Format: 
    # {0: 0, 1: 1, ...} # a dict of userid:nodeid
    # hashtag1
    # [user11, user12, ...]
    os.makedirs(outputfile_dirpath, exist_ok=True)
    with open(os.path.join(outputfile_dirpath, ce_init_diffusion_dict_filename), "w") as f:
        f.write(str(dict(
            (str(i), i) for i in range(graph_vcount)
        )) + '\n')
        cnt = 0
        for hashtag, cascade_list in diffusion_dict.items():
            if len(cascade_list) < min_influence or len(cascade_list) > max_influence:
                continue
            cnt += 1
            f.write(str(hashtag) + '\n')
            f.write(str([str(item[0]) for item in cascade_list]) + '\n')
    logger.info(f"Save actionlog in Format2 with {cnt} Diffusions")

def read_actionlog():
    diffusion_dict = {}
    with open(os.path.join(outputfile_dirpath, "sorted_actionlog"), "r") as f:
        for line in f:
            item_list = line[:-1].split(' ')
            hashtag = item_list[0]
            if len(item_list[1:]) < min_influence or len(item_list[1:]) > max_influence:
                continue
            user_timestamp_list = []
            for item in item_list[1:]:
                user, timestamp = item.split(',')
                user_timestamp_list.append((int(user), int(timestamp)))
            diffusion_dict[hashtag] = user_timestamp_list
    logger.info(f"Read actionlog with {len(diffusion_dict)} Diffusions")
    return diffusion_dict

def summarize_diffusion(diffusion_dict: Dict[str, List]):
    diffusion_size = [len(v) for v in diffusion_dict.values()]
    logger.info("mean diffusion length %.2f", np.mean(diffusion_size))
    logger.info("max diffusion length %.2f", np.max(diffusion_size))
    logger.info("min diffusion length %.2f", np.min(diffusion_size))
    for i in range(1, 10):
        logger.info("%d-th percentile of diffusion length %.2f", i*10, np.percentile(diffusion_size, i*10))
    logger.info("95-th percentile of diffusion length %.2f", np.percentile(diffusion_size, 95))

def summarize_graph(graph):
    degree = graph.degree()
    logger.info("mean degree %.2f", np.mean(degree))
    logger.info("max degree %.2f", np.max(degree))
    logger.info("min degree %.2f", np.min(degree))
    for i in range(1, 10):
        logger.info("%d-th percentile of degree %.2f", i*10, np.percentile(degree, i*10))
    logger.info("95-th percentile of degree %.2f", np.percentile(degree, 95))

def summarize_distribution(data: List[Any]):
    logger.info(f"summarize distribution for list with length of {len(data)}")
    logger.info("mean list length %.2f", np.mean(data))
    logger.info("max list length %.2f", np.max(data))
    logger.info("min list length %.2f", np.min(data))
    for i in range(1, 10):
        logger.info("%d-th percentile of list length %.2f", i*10, np.percentile(data, i*10))

def get_active_users(diffusion_dict: Dict[str, Any]):
    active_users_set = set()
    for _, cascade in diffusion_dict.items():
        # if len(cascade) < min_influence or len(cascade) > max_influence:
        if len(cascade) < 10:
            continue
        active_users_set |= set([item[0] for item in cascade])
    
    logger.info(f"Total {len(active_users_set)} Active Users")
    return active_users_set

def get_activers_and_candidates(diffusion_dict: CASCADES_TYPE, graph) -> Tuple[set, set]:
    active_users_set, candidates_set = set(), set()
    for _, cascade in diffusion_dict.items():
        for user, _ in cascade:
            active_users_set |= set([user])
            candidates_set   |= set(graph.neighbors(user))
    
    # NOTE: candidates-=active_user_set
    candidates_set -= active_users_set

    logger.info(f"Total {len(active_users_set)} Active Users, and {len(candidates_set)} Candidates")
    return active_users_set, candidates_set

def stable_sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig
