import networkx as nx
import random
import scipy.stats as ss

theta1, theta2 = 0.2, 0.8

# edgelist_filepath = "higgs/higgs-social_network.edgelist"
edgelist_filepath = "munmun_twitter_social/munmun_twitter_social/out.munmun_twitter_social"

# genactionlog_filepath = "higgs-output/gen-actionlog-300.txt"
genactionlog_filepath = "munmun_twitter_social-output/gen-actionlog-300.txt"

# NOTE: 
# TODO: 为什么Weight计算公式是线性函数, 是否需要额外处理使其满足Sigma<=1的条件
# graph-seq         : 用于标识影响轮次, 也即timestamp
# node-threshold    : 用于标识节点的阈值
# node-influencesum : 用于标识活跃节点对其的影响
# edge-weight       : 用于标识边权
def read_network(hashtag, filename):
    G = nx.DiGraph(hashtag = hashtag)
    with open(filename, 'r') as f:
        for line in f:
            if line[0] == '%':
                continue
            from_, to_ = line[:-1].split(' ')[:2]
            if from_ not in G.nodes():
                G.add_node(
                    from_,
                    threshold = random.uniform(0,1),
                    influencesum = 0
                )
            if to_ not in G.nodes():
                G.add_node(
                    to_,
                    threshold = random.uniform(0,1),
                    influencesum = 0
                )
            G.add_edge(from_, to_)

    # compute sum of all its incoming node's out-degree for each node, 
    out_degree_sum_map = {}
    for u in G.nodes():
        sum = 0
        for in_node, _ in G.in_edges(u):
            sum += G.out_degree(in_node)
        out_degree_sum_map[u] = sum
    
    for (u, v) in G.edges:
        G.edges[u, v]["weight"] = float(theta1 + theta2 * G.out_degree(u)) / out_degree_sum_map[v]
    return G

# 用于生成符合特定分布的随机数, 这里的分布是正态分布
# 主流的实现方法有ITM和ARM两种, 这里实现的是较为简单且不需要CDF函数的ARM方法
# 说明: x模拟0~1之间的均匀分布, y模拟0~0.5间的均匀分布, 如果y落在F(x)正态曲线下方, 则说明此时成功生成了一组符合正态分布的随机数
# 参考链接: https://blog.codinglabs.org/articles/methods-for-generating-random-number-distributions.html
def standard_normal_rand():
    while True:
        x = random.uniform(0, 1)
        y = random.uniform(0, 0.5)
        if y < ss.norm.pdf(x, loc=0.5):
            return x

def init_seed_node(graph, nodes, num=1):
    seed_node_list = []
    for _ in range(num):
        seed_node = random.choice(nodes)
        while len(list(graph.neighbors(seed_node))) == 0:
            seed_node = random.choice(nodes)
        seed_node_list.append(seed_node)
    return seed_node_list

def gen_actionlog(G, param={
    "max_active_user": 2000,
    "max_timestamp": 1200
}):
    actionlog = []
    active_user = {}
    seed_node_list = init_seed_node(G, list(G.nodes))
    timestamp = 0

    seed_node = seed_node_list[0]
    actionlog.append((
        seed_node,
        str(G.graph["hashtag"]),
        timestamp
    ))
    active_user[seed_node] = 0
    timestamp += 1

    while len(active_user) <= param["max_active_user"] and timestamp <= param["max_timestamp"]:
        random_active_user = random.choice(list(active_user.keys()))
        for nonactive_user in G.neighbors(random_active_user):
            if nonactive_user in active_user:
                continue
            if G.has_edge(random_active_user, nonactive_user):
                prob = standard_normal_rand()
                if random.uniform(0,1) < prob:
                    G.nodes[nonactive_user]["influencesum"] += G.edges[random_active_user, nonactive_user]["weight"]
            if G.nodes[nonactive_user]["influencesum"] >= G.nodes[nonactive_user]["threshold"]:
                actionlog.append((
                    nonactive_user,
                    str(G.graph["hashtag"]),
                    timestamp
                ))
                active_user[nonactive_user] = 1
        # early stop
        if len(active_user) == 1:
            break

        timestamp += 1
    
    return actionlog, len(active_user), timestamp

g = read_network(hashtag=1, filename=edgelist_filepath)

for hashtag in range(1, 301):
    g.graph["hashtag"] = hashtag
    actionlog, active_user_num, last_timestamp = gen_actionlog(G=g)
    print(f"len(actionlog): {len(actionlog)}, active_user_num: {active_user_num}, last_timestamp: {last_timestamp}")

    with open(genactionlog_filepath, 'a', encoding='utf8') as f:
        for user, hashtag, timestamp in actionlog:
            f.write(f"{user} {hashtag} {timestamp}\n")
