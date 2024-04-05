import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from util import *
import random
import itertools

# 1. Read Network and ActionLog
# graph = read_network()
# diffusion_dict = read_actionlog()

def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

graph = load_pickle("/root/Lab_Related/data/Heter-GAT/Classic/graph/graph-undirected.p")
diffusion_dict = load_pickle("/root/Lab_Related/data/Heter-GAT/Classic/ActionLog.p")

# 2. Gen Subnetwork Sample
class PreprocessData:
    def __init__(self):
        self.adj_matrices = []
        self.influence_features = []
        self.vertex_ids = []
        self.labels = []
    def __len__(self):
        return len(self.labels)

def random_walk_with_restart(g, start, restart_prob):
    # start is list
    current = random.choice(start)
    stop = False
    while not stop:
        stop = yield current
        current = random.choice(start) if random.random() < restart_prob or g.degree(current)==0 \
                else random.choice(g.neighbors(current))

def create_sample(user, label, user_affected_now, gensample):
    active_neighbor, inactive_neighbor = [], []
    for v in graph.neighbors(user):
        if v in user_affected_now:
            active_neighbor.append(v)
        else:
            inactive_neighbor.append(v)
    if len(active_neighbor) < min_active_neighbor:
        # logger.info(f"ERR[len(active_neighbor)<3] len(active_neighbor): {len(active_neighbor)}")
        return
    subnetwork_size = ego_size + 1
    subnetwork = []
    if len(active_neighbor) < ego_size:
        # include some inactive_neighbor
        subnetwork = set(active_neighbor)
        for v in itertools.islice(
            random_walk_with_restart(
                graph,
                start=active_neighbor + [user,],
                restart_prob=restart_prob,
            ),
            walk_length
        ):
            if v != user and v not in subnetwork:
                subnetwork.add(v)
                if len(subnetwork) == ego_size:
                    break
        subnetwork = list(subnetwork)
        if len(subnetwork) < ego_size:
            # logger.info(f"ERR[len(subnetwork)<49] len(sub-network): {len(subnetwork)}")
            return
    else:
        samples = np.random.choice(
            active_neighbor,
            size=ego_size,
            replace=False
        )
        subnetwork = samples.tolist()
    subnetwork.append(user)

    ranks = np.array(subnetwork).argsort().argsort()
    subgraph = graph.subgraph(subnetwork, implementation="create_from_scratch")
    adjacency = np.array(subgraph.get_adjacency().data, dtype=int)
    # convert ordered adj-matrix to an original one
    # i.e. subnetwork = [12,5,13], and we have the corresponding adj-matrix A
    # but A is an ordered one, which A[0][2] means edges between 5 and 13, not 12 and 13
    # so we need to convert it with original ranks
    # thus we use [][:,] to convert it in both row and column directions
    adjacency = adjacency[ranks][:,ranks]
    gensample.adj_matrices.append(adjacency)

    influence_feature = np.zeros((subnetwork_size,2))
    for idx, v in enumerate(subnetwork[:-1]):
        if v in user_affected_now:
            influence_feature[idx, 0] = 1
    influence_feature[subnetwork_size-1,1] = 1
    gensample.influence_features.append(influence_feature)

    gensample.vertex_ids.append(np.array(subnetwork, dtype=int))

    gensample.labels.append(label)

def dump_preprocess_data(desc, adj_matrices, influence_features, vertex_ids, labels):
    os.makedirs(desc, exist_ok=True)
    adj_matrices = np.array(adj_matrices)
    influence_features = np.array(influence_features)
    vertex_ids = np.array(vertex_ids)
    labels = np.array(labels)

    with open(f"{desc}/adjacency_matrix.npy", "wb") as f:
        np.save(f, adj_matrices)
    with open(f"{desc}/influence_feature.npy", "wb") as f:
        np.save(f, influence_features)
    with open(f"{desc}/vertex_id.npy", "wb") as f:
        np.save(f, vertex_ids)
    with open(f"{desc}/label.npy", "wb") as f:
        np.save(f, labels)

    logger.info("Dump %d instances in total" % (len(labels)))

# 2.

degree = graph.degree()
Nslice = Ntimeslice + 1
gensample_list = [PreprocessData() for _ in range(Nslice)] # gensample_list[0] is None

for idx, (hashtag, cascades) in enumerate(diffusion_dict.items()):
    if idx < 0:
        continue

    cascade_idx = 0
    prev_pos, prev_neg = set(), set()
    user_affected_now = set()
    min_timestamp, max_timestamp = cascades[0][1], cascades[-1][1]
    time_span = (max_timestamp-min_timestamp)/Nslice

    for tidx in range(Nslice):
        lower_b, upper_b = min_timestamp+time_span*tidx, min_timestamp+time_span*(tidx+1)
        cur_pos, cur_neg = set(), set()

        while cascade_idx < len(cascades) and (cascades[cascade_idx][1] >= lower_b and cascades[cascade_idx][1] <= upper_b):
            if cascades[cascade_idx][0] not in user_affected_now:
                cur_pos.add(cascades[cascade_idx][0])
                cur_neg |= set(graph.neighbors(cascades[cascade_idx][0]))
            cascade_idx += 1
        
        if tidx > 0:
            for active_user in prev_pos:
                if degree[active_user] >= min_degree and degree[active_user] <= max_degree:
                    create_sample(user=active_user, label=1, user_affected_now=user_affected_now, gensample=gensample_list[tidx])
            
            neg_user = prev_neg - cur_pos
            sample_neg_user = np.random.choice(list(neg_user), size=min(negative_sample_num * len(prev_pos), len(neg_user)), replace=False)
            for inactive_user in sample_neg_user:
                if degree[inactive_user] >= min_degree and degree[inactive_user] <= max_degree:
                    create_sample(user=inactive_user, label=0, user_affected_now=user_affected_now, gensample=gensample_list[tidx])
        
        user_affected_now |= cur_pos
        prev_pos, prev_neg = cur_pos, cur_neg
        # logger.info(f"len(user_affected_now): {len(user_affected_now)}, len(prev_pos): {len(prev_pos)}, len(prev_neg): {len(prev_neg)}")

    logger.info(f"idx: {idx:>6}, hashtag: {hashtag:>6}, cascades length: {len(cascades):>6}, sample_l length:" + \
        " ".join([f"{len(sample):>6}" for sample in gensample_list[1:]])
    )

for idx in range(1, Nslice):
    dump_preprocess_data(
        desc=f"stages/{idx}",
        adj_matrices=gensample_list[idx].adj_matrices,
        influence_features=gensample_list[idx].influence_features,
        vertex_ids=gensample_list[idx].vertex_ids,
        labels=gensample_list[idx].labels
    )
