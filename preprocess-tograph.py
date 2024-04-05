import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from util import *

# 1. Build Network
# Suppose Given Graph is in EdgeList Form
node2graphid_map, edgelist = build_network_from_edgelist()
save_network(node2graphid_map=node2graphid_map, edgelist=edgelist)
graph = init_graph(node2graphid_map=node2graphid_map, edgelist=edgelist, save_graph=True)

# 2. Build ActionLog
diffusion_dict = build_actionlog(node2graphid_map=node2graphid_map)
save_actionlog(diffusion_dict=diffusion_dict)

# 3. Summarize Graph and DiffusionDict
summarize_diffusion(diffusion_dict=diffusion_dict)
summarize_graph(graph=graph)

# 4-1. Compute Vertex Feature For DeepInf-Model
def compute_structural_features(graph):
    logger.info("Computing rarity (reciprocal of degree)")
    degree = np.array(graph.degree())
    degree[degree==0] = 1
    rarity = 1. / degree
    logger.info("Computing clustering coefficient..")
    cc = graph.transitivity_local_undirected(mode="zero")
    logger.info("Computing pagerank...")
    pagerank = graph.pagerank(directed=False)
    logger.info("Computing authority_score...")
    authority_score = graph.authority_score()
    logger.info("Computing hub_score...")
    hub_score = graph.hub_score()
    logger.info("Computing evcent...")
    evcent = graph.evcent(directed=False)
    logger.info("Computing coreness...")
    coreness = graph.coreness()
    logger.info("Structural feature computation done!")
    structural_features = np.column_stack(
        (rarity, cc, pagerank,
            #constraint, closeness, betweenness,
            authority_score, hub_score, evcent, coreness)
    )
    
    os.makedirs(f"{outputfile_dirpath}/feature", exist_ok=True)
    with open(os.path.join(outputfile_dirpath, "feature/vertex_feature.npy"), "wb") as f:
        np.save(f, structural_features)

compute_structural_features(graph=graph)

# 4-2. Summarize Active-Neighbor-Num Distribution

# minus fake nodes
graph_vcount = graph.vcount() - ego_size

def summarize_active_neighbor_num_distribution(graph, diffusion_dict: Dict[str, Any], graph_vcount):
    active_users = get_active_users(diffusion_dict=diffusion_dict)

    # count all nodes' active_neighbor_num
    active_neighbor_num_list = []
    for node in range(0, graph_vcount):
        active_neighbor_num = 0
        for v in graph.neighbors(node):
            if v in active_users:
                active_neighbor_num += 1
        active_neighbor_num_list.append(active_neighbor_num)
    
    summarize_distribution(data=active_neighbor_num_list)

summarize_active_neighbor_num_distribution(graph=graph, diffusion_dict=diffusion_dict, graph_vcount=graph_vcount)

# 4-3. Cascade Embedding Feature
# Generate Some Pre-File For CE-Feature Calculation
save_actionlog2(diffusion_dict=diffusion_dict, graph_vcount=graph_vcount)
