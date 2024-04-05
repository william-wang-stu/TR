import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from util import *
from scipy.stats import rankdata
import re
import math

# Virality2013
min_degree, max_degree, min_active_neighbor = 3, 200, 3
min_influence, max_influence = 30, 200
outputfile_dirpath = "../output/data/virality2013-new-output"
edgelist_filepath = os.path.join(source_data_dir, "virality2013/follower_gcc.anony.dat")
actionlog_filepath = os.path.join(source_data_dir, "virality2013-output/actionlog.txt")
cascade_embedding_matrix_filepath = "../cascade-embedding/data/cascade-embedding/virality2013-cascade_embedding.npy"

# Munmun
# min_degree, max_degree, min_active_neighbor = 1, 700, 2
# min_influence, max_influence = 0, 10000
# outputfile_dirpath = "../output/data/munmun-new-output"
# edgelist_filepath = os.path.join(source_data_dir, "munmun_twitter_social/munmun_twitter_social.edges")
# actionlog_filepath = os.path.join(source_data_dir, "munmun_twitter_social-output/gen-actionlog-300.txt")
# cascade_embedding_matrix_filepath = "../cascade-embedding/data/cascade-embedding/munmun-cascade_embedding.npy"

# Higgs
# min_degree, max_degree, min_active_neighbor = 3, 200, 3
# min_influence, max_influence = 0, 10000
# outputfile_dirpath = "../output/data/higgs-new-output"
# edgelist_filepath  = os.path.join(source_data_dir, "higgs/higgs-social_network.edgelist")
# actionlog_filepath = os.path.join(source_data_dir, "higgs-output/gen-actionlog-300.txt")
# cascade_embedding_matrix_filepath = "../cascade-embedding/data/cascade-embedding/higgs-cascade_embedding.npy"

# 1. Read Network and ActionLog
def read_network2(is_directed=False):
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

def read_actionlog2():
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

graph = read_network2()
diffusion_dict = read_actionlog2()

# graph = read_network()
# diffusion_dict = read_actionlog()

# 1-2. Calculate Common Vars
# NOTE: 每个stage的active_user_set和diffusion_dict都不相同
graph_vcount = graph.vcount() - ego_size
Nslice = Ntimeslice + 1

# 1-2-2. Hyper Param
# tau, net = 2, 10000
# alpha, beta = 0.8, 0.1
# gamma = 1-alpha-beta
# pos_percent = 0.1

# activated_user_num = math.ceil(pos_percent * graph_vcount)

# 1-3. Get Active Users and Candidates
# def split_diffusion_dict(cascades: CASCADES_TYPE, Nstages=Nslice) -> List[CASCADES_TYPE]:
#     cascades_l: CASCADES_TYPE = [{} for _ in range(Nstages)]

#     for hashtag, cascade in cascades.items():
#         for tidx in range(Nstages):
#             cascades_l[tidx][hashtag] = []
        
#         cascade_elem_idx = 0
#         min_t, max_t = cascade[0][1], cascade[len(cascade)-1][1]
#         diff_t = (max_t - min_t) / Nstages

#         lower_bound, upper_bound = min_t, min_t + diff_t
#         for tidx in range(Nstages):
#             while cascade_elem_idx < len(cascade) and cascade[cascade_elem_idx][1] >= lower_bound and cascade[cascade_elem_idx][1] <= upper_bound:
#                 cascades_l[tidx][hashtag].append((cascade[cascade_elem_idx][0], cascade[cascade_elem_idx][1]))
#                 cascade_elem_idx += 1
#             lower_bound += diff_t
#             upper_bound += diff_t

#     return cascades_l

# splitted_cascades = split_diffusion_dict(diffusion_dict)

def read_cascades(tidx, split_idx):
    cascades = {}
    with open(os.path.join("data", f"cascades_kfold{split_idx}_stage{tidx}.txt"), "r") as f:
        for line in f:
            item_list = line[:-1].split(' ')
            hashtag = item_list[0]
            user_timestamp_list = []
            for item in item_list[1:]:
                user, timestamp = item.split(',')
                user_timestamp_list.append((int(user), int(timestamp)))
            cascades[hashtag] = user_timestamp_list
    return cascades

# KFold Cross-Validation
SPLIT_NUM = 5
kf_cascades = [[{} for _ in range(Nslice)] for _ in range(SPLIT_NUM)]
for split_idx in range(SPLIT_NUM):
    for tidx in range(Nslice):
        kf_cascades[split_idx][tidx] = read_cascades(tidx, split_idx)

# 1-4. Gen Random Diffusion Dict
# random_diffusion_dict = {}
# random_elem_num = 300

# random_diffusion_keys = np.random.choice(list(diffusion_dict.keys()), size=random_elem_num, replace=False)
# for random_key in random_diffusion_keys:
#     random_diffusion_dict[random_key] = diffusion_dict[random_key]

# with open(os.path.join(outputfile_dirpath, "random_diffusion_dict"), "w") as f:
#     for key, value in random_diffusion_dict.items():
#         f.write(f"{key}")
#         for user, timestamp in value:
#             f.write(f" {user},{timestamp}")
#         f.write("\n")
# logger.info("Write Random Diffusion Dict with Length %d to File %s" % (random_elem_num, os.path.join(outputfile_dirpath, "random_diffusion_dict")))

# 2-1. Calculate Gravity Feature: 1 * |V-Candidates|
def get_gravity_feature(graph, active_users, order_t, tao_t):
    gravity_feature = [0 for _ in range(graph_vcount)]
    beta = 1.0 # no affect to ranking result, 因为大家都乘系数等于大家都不乘

    cnt = 0
    for active_user in active_users:
        cnt += 1
        if cnt % 1000 == 0:
            logger.info(f"Calculated {cnt} Active Users in Gravity Feature")
        center_user_set = set([active_user])
        for order_param in range(1, order_t+1):
            next_level_user_set = set()
            for center_user in center_user_set:
                for neighbor_user in graph.neighbors(center_user):
                    next_level_user_set.add(neighbor_user)
                    # if neighbor_user in candidates:
                    gravity_feature[neighbor_user] += beta * graph.degree(active_user) * graph.degree(neighbor_user) / pow(order_param, tao_t)
            # logger.info(f"cur level size: {len(center_user_set)}, next level size: {len(next_level_user_set)}")
            center_user_set = next_level_user_set
    
    return gravity_feature

# os.makedirs(f"{outputfile_dirpath}/feature_stages", exist_ok=True)
# # for tau in [1,2,3]:
# tau = 4
# for tidx in range(Ntimeslice):
#     gravity_feature = get_gravity_feature(
#         graph=graph,
#         active_users=activers_l[tidx],
#         order_t=2,
#         tao_t=tau,
#     )

#     with open(os.path.join(outputfile_dirpath, f"feature_stages/gravity_feature_tau{tau}_stage{tidx}.npy"), "wb") as f:
#         np.save(f, gravity_feature)
#     logger.info(f"Calculated Gravity Feature with hyper-param tao={tau}, stages={tidx}")

# 2-2. Exposure Time Feature: 1 * |V-Candidates|
def get_exposure_time_sum_distribution(diffusion_dict: Dict[str, Any], graph, BUCKET_NUM):
    exposure_time_sum = [0 for _ in range(graph_vcount)]

    # cnt = 0
    for _, cascade in diffusion_dict.items():
        # cnt += 1
        # if cnt % 1000 == 0:
        #     logger.info(f"Calculated {cnt} cascades in Exposure Time Sum Distribution")
    
        for user, timestamp in cascade[1:]:
            cascade_elem_idx = 0
            while cascade_elem_idx < len(cascade) and cascade[cascade_elem_idx][1] < timestamp:
                if user == cascade[cascade_elem_idx][0]:
                    cascade_elem_idx += 1
                    continue
                if cascade[cascade_elem_idx][0] in graph.neighbors(user):
                    exposure_time_sum[user] += timestamp - cascade[cascade_elem_idx][1]
                cascade_elem_idx += 1
        
    logger.info(f"exposure_time_sum elem>0 number: {len(list(filter(lambda item: item>0, exposure_time_sum)))}")

    if len(list(filter(lambda item: item>0, exposure_time_sum))) == 0:
        logger.info("Error: exposure_time_sum is straight zeros")
        return [], -1

    # split exposure_num into several buckets
    min_et_sum, max_et_sum = min(exposure_time_sum), max(exposure_time_sum)
    # logger.info(f"min_exposure_num: {min_et_sum}, max_exposure_num: {max_et_sum}")

    bucket_range = (max_et_sum - min_et_sum) / BUCKET_NUM
    buckets = [0 for _ in range(BUCKET_NUM + 1)]
    for et_sum in exposure_time_sum:
        if et_sum > 0:
            bucket_idx = round((et_sum - min_et_sum) / bucket_range)
            buckets[bucket_idx] += 1
            # logger.info(bucket_idx, et_sum)
    # logger.info(f"buckets: {buckets}")

    return buckets, bucket_range

def get_timestamp_upper_bound(timestamp, time_lower_bound, time_upper_bound, Nslice=Ntimeslice):
    time_range = (time_upper_bound - time_lower_bound) / Nslice + 1e-4
    time_idx = int((timestamp - time_lower_bound) / time_range) + 1
    return min(time_upper_bound, time_idx * time_range + time_lower_bound)

def get_exposure_time_feature(buckets, bucket_range, graph, diffusion_dict: CASCADES_TYPE, BUCKET_NUM):
    exposure_time_feature = [0 for _ in range(graph_vcount)]
    buckets_sum = sum(buckets)

    cnt = 0
    for _, cascade in diffusion_dict.items():
        if len(cascade) == 0:
            continue
        cnt += 1
        if cnt % 10 == 0:
            logger.info(f"Calculated {cnt} cascades in Exposure Time Feature")
        
        lower_bound, upper_bound = cascade[0][1], cascade[len(cascade)-1][1]
        for active_user, timestamp in cascade:
            timestamp_upper_bound = get_timestamp_upper_bound(timestamp=timestamp, time_lower_bound=lower_bound, time_upper_bound=upper_bound)
            for target_user in graph.neighbors(active_user):
                exposure_time_feature[target_user] += timestamp_upper_bound - timestamp
    
    for etf_idx, etf_elem in enumerate(exposure_time_feature):
        if etf_elem > 0:
            bucket_idx = round( (etf_elem - 0) / bucket_range)
            if bucket_idx > BUCKET_NUM:
                bucket_idx = BUCKET_NUM
            exposure_time_feature[etf_idx] = buckets[bucket_idx] / buckets_sum

    logger.info(f"exposure_time_feature non-zero distribution: {np.unique(list(filter(lambda item:item>0, exposure_time_feature)), return_counts=True)}")

    return exposure_time_feature

# BUCKET_NUM = 10000
# # for BUCKET_NUM in [1000, 10000, 100000]:
# for tidx in range(Ntimeslice):
#     buckets, bucket_range = get_exposure_time_sum_distribution(
#         diffusion_dict=splitted_cascades[tidx],
#         graph=graph,
#         BUCKET_NUM=BUCKET_NUM
#     )
#     exposure_time_feature = get_exposure_time_feature(
#         buckets=buckets,
#         bucket_range=bucket_range,
#         graph=graph,
#         diffusion_dict=splitted_cascades[tidx],
#         BUCKET_NUM=BUCKET_NUM
#     )

#     with open(os.path.join(outputfile_dirpath, f"feature_stages/exposure_time_feature_net{BUCKET_NUM}_stage{tidx}.npy"), "wb") as f:
#         np.save(f, exposure_time_feature)
#     logger.info(f"Calculated Exposure Time Feature with hyper-param Net={BUCKET_NUM}, stage={tidx}")

# 2-3. Cascade Embedding Feature: 1 * |V-Candidates|
# Generate CE-Vector with Skip-Gram, Results are in cascade_embedding sub-directory

# ce_vec_dict = {}
# vocab_size, embed_size = 0, 0
# with open(cascade_embedding_matrix_filepath, 'r') as f:
#     for idx, line in enumerate(f):
#         if idx == 0:
#             vocab_size, embed_size = line.split(' ')
#         else:
#             idx, emb = line.split(' ', 1)
#             emb = re.sub('\s|\t|\n|"|\'', '', emb[1:-2])
#             emb = [float(elem) for elem in emb.split(',')]
#             ce_vec_dict[int(idx)] = emb
# logger.info("Read Cascade Embedding Matrix")

# # Normalize Each Vector
# norm_ce_vec_dict = {}
# for key, value in ce_vec_dict.items():
#     norm_ce_vec_dict[key] = value / np.linalg.norm(value)

# def get_cascade_embedding_feature(active_users, graph_vcount):
#     cascade_embedding_feature = [0 for _ in range(graph_vcount)]

#     ce_matrix = np.array([norm_ce_vec_dict[active_user] for active_user in active_users]).reshape(-1,128)
#     active_user_num = ce_matrix.shape[0]

#     for user in range(graph_vcount):
#         intermediate_result = np.matmul(ce_matrix, norm_ce_vec_dict[user])
#         cascade_embedding_feature[user] = active_user_num - intermediate_result.sum()
    
#     return cascade_embedding_feature

# for tidx in range(Ntimeslice):
#     cascade_embedding_feature = get_cascade_embedding_feature(
#         active_users=(activers_l[tidx]),
#         graph_vcount=graph_vcount
#     )

#     with open(os.path.join(outputfile_dirpath, f"feature_stages/cascade_embedding_feature_stage{tidx}.npy"), "wb") as f:
#         np.save(f, cascade_embedding_feature)
#     logger.info(f"Calculated Cascade Embedding Feature with stages={tidx}")

# Cross Validation
for split_idx in range(SPLIT_NUM):
    split_cascades = kf_cascades[split_idx]

    activers_l, candidates_l = [set() for _ in range(Nslice)], [set() for _ in range(Nslice)]
    for tidx in range(Nslice):
        activers_l[tidx], candidates_l[tidx] = get_activers_and_candidates(split_cascades[tidx], graph=graph)

    # Gravity Feature
    os.makedirs(f"{outputfile_dirpath}/feature_cross_validation", exist_ok=True)
    # for tau in [1,2,3]:
    tau = 0
    for tidx in range(Ntimeslice):
        gravity_feature = get_gravity_feature(
            graph=graph,
            active_users=activers_l[tidx],
            order_t=2,
            tao_t=tau,
        )

        with open(os.path.join(outputfile_dirpath, f"feature_cross_validation/gravity_feature_tau{tau}_stage{tidx}_kfold{split_idx}.npy"), "wb") as f:
            np.save(f, gravity_feature)
        logger.info(f"Calculated Gravity Feature with hyper-param tao={tau}, stages={tidx}, kfold={split_idx}")

    # # Exposure Time Feature
    # BUCKET_NUM = 10000
    # # for BUCKET_NUM in [1000, 10000, 100000]:
    # for tidx in range(Ntimeslice):
    #     buckets, bucket_range = get_exposure_time_sum_distribution(
    #         diffusion_dict=split_cascades[tidx],
    #         graph=graph,
    #         BUCKET_NUM=BUCKET_NUM
    #     )
    #     exposure_time_feature = get_exposure_time_feature(
    #         buckets=buckets,
    #         bucket_range=bucket_range,
    #         graph=graph,
    #         diffusion_dict=split_cascades[tidx],
    #         BUCKET_NUM=BUCKET_NUM
    #     )

    #     with open(os.path.join(outputfile_dirpath, f"feature_cross_validation/exposure_time_feature_net{BUCKET_NUM}_stage{tidx}_kfold{split_idx}.npy"), "wb") as f:
    #         np.save(f, exposure_time_feature)
    #     logger.info(f"Calculated Exposure Time Feature with hyper-param Net={BUCKET_NUM}, stage={tidx}, kfold={split_idx}")

    # # Cascade Embedding Feature
    # for tidx in range(Ntimeslice):
    #     cascade_embedding_feature = get_cascade_embedding_feature(
    #         active_users=(activers_l[tidx]),
    #         graph_vcount=graph_vcount
    #     )

    #     with open(os.path.join(outputfile_dirpath, f"feature_cross_validation/cascade_embedding_feature_stage{tidx}_kfold{split_idx}.npy"), "wb") as f:
    #         np.save(f, cascade_embedding_feature)
    #     logger.info(f"Calculated Cascade Embedding Feature with stages={tidx}, kfold={split_idx}")


# def cal_ypred(gravity_ranking, exposure_time_ranking, cascade_embedding_ranking, candidates):
#     triple_ranking = alpha * gravity_ranking + beta * exposure_time_ranking + gamma * cascade_embedding_ranking

#     cnt = 0
#     y_pred = [0 for _ in range(graph_vcount)]
#     # NOTE: triple_ranking_val是融合后的排名，目标是从中找出前activated_user_num的用户ID
#     for idx in np.argsort(triple_ranking):
#         if cnt == activated_user_num:
#             break
#         if idx in candidates:
#             cnt += 1
#             y_pred[idx] = 1
#     return y_pred

# def cal_ytrue(active_users):
#     y_true = [0 for _ in range(graph_vcount)]
#     for active_user in active_users:
#         y_true[active_user] = 1
#     return y_true

# def set_tp_fn_fp_tn(y_pred, y_true):
#     true_pos, false_neg, false_pos, true_neg = 0, 0, 0, 0
#     for y_pred_elem, y_true_elem in zip(y_pred, y_true):
#         true_pos  += ( y_pred_elem == 1 and y_true_elem == 1)
#         false_neg += ( y_pred_elem == 0 and y_true_elem == 1)
#         false_pos += ( y_pred_elem == 1 and y_true_elem == 0)
#         true_neg  += ( y_pred_elem == 0 and y_true_elem == 0)
#     # logger.info(f"tp: {true_pos}, fn: {false_neg}, fp: {false_pos}, tn: {true_neg}")
#     return true_pos, false_neg, false_pos, true_neg

# def cal_measure(true_pos, false_neg, false_pos, true_neg):
#     prec = true_pos / (true_pos + false_pos)
#     rec  = true_pos / (true_pos + false_neg)
#     f1   = 2 * prec * rec / (prec + rec)
#     logger.info(f"Prec. : {prec:>4}, Rec. : {rec:>4}, F1: {f1:>4}")

# def get_rank(elem_list):
#     return rankdata(-elem_list, method='min')

# tp_l, fn_l, fp_l, tn_l = [0 for _ in range(Ntimeslice)], [0 for _ in range(Ntimeslice)], [0 for _ in range(Ntimeslice)], [0 for _ in range(Ntimeslice)]

# for tidx in range(Ntimeslice):
#     gravity_ranking = get_rank(np.load(os.path.join(outputfile_dirpath, f"feature_stages/gravity_feature_tau2_stage{tidx}.npy")))
#     exposure_time_ranking = get_rank(np.load(os.path.join(outputfile_dirpath, f"feature_stages/exposure_time_feature_net10000_stage{tidx}.npy")))
#     cascade_embedding_ranking = get_rank(np.load(os.path.join(outputfile_dirpath, f"feature_stages/cascade_embedding_feature_stage{tidx}.npy")))

#     y_pred = cal_ypred(
#         gravity_ranking=gravity_ranking, 
#         exposure_time_ranking=exposure_time_ranking, 
#         cascade_embedding_ranking=cascade_embedding_ranking,
#         candidates=candidates_l[tidx]
#     )
#     y_true = cal_ytrue(active_users=activers_l[tidx+1])
#     tp, fn, fp, tn = set_tp_fn_fp_tn(y_pred, y_true)

#     tp_l[tidx] += tp
#     fn_l[tidx] += fn
#     fp_l[tidx] += fp
#     tn_l[tidx] += tn
#     logger.info(f"tp_l={tp_l}, fn_l={fn_l}, fp_l={fp_l}, tn_l={tn_l}")

# for tidx in range(Ntimeslice):
#     cal_measure(tp_l[tidx], fn_l[tidx], fp_l[tidx], tn_l[tidx])
