import os

# hyper-param
ego_size = 49 # without center node, which means containing ego_size + 1 nodes per batch
Ntimeslice = 8
Nslice = Ntimeslice + 1
negative_sample_num = 1
restart_prob = 0.2
walk_length = 1000

# filedir config
ce_init_diffusion_dict_filename = "ce_init_diffusion_dict.npy"

# source_data_dir = "/root/TR-pptusn/pptusn-3dataset/"
source_data_dir = "/root/Lab_Related/data/Heter-GAT/Classic"

# Lab-Data
min_degree, max_degree, min_active_neighbor = 3, 14715, 3
# min_influence, max_influence = 10, 200
outputfile_dirpath = "/root/Heter-GAT/src/user_features"
# edgelist_filepath = os.path.join(source_data_dir, "graph/U-U.p")
edgelist_filepath = os.path.join(source_data_dir, "graph/edgelist-added.txt")
actionlog_filepath = os.path.join(source_data_dir, "ActionLog.p")
cascade_embedding_matrix_filepath = "../cascade-embedding/data/cascade-embedding/virality2013-cascade_embedding.npy"

# Virality2013
# min_degree, max_degree, min_active_neighbor = 3, 200, 3
# min_influence, max_influence = 30, 200
# outputfile_dirpath = "../output/data/virality2013-new-output"
# edgelist_filepath = os.path.join(source_data_dir, "virality2013/follower_gcc.anony.dat")
# actionlog_filepath = os.path.join(source_data_dir, "virality2013-output/actionlog.txt")
# cascade_embedding_matrix_filepath = "../cascade-embedding/data/cascade-embedding/virality2013-cascade_embedding.npy"

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
