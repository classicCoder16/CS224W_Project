import snap
import math
import numpy as np
import scipy

#####################################
######LINK PREDICTION MEASURES#######
#####################################

def get_2_hops(G, n1, n2, directed=False):
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True
	result = 0
	if (n1 >= 18630353 and n1 < 31097109) and (n2 < 18630353 or n2 >= 31097109):
		b_neighbors = snap.TIntV()
		snap.GetNodesAtHop(G, n1, 1, b_neighbors, directed)
		for n in b_neighbors:
			result += snap.GetCmnNbrs(G, n, n2)
	elif (n2 >= 18630353 and n2 < 31097109) and (n1 < 18630353 or n1 >= 31097109):
		b_neighbors = snap.TIntV()
		snap.GetNodesAtHop(G, n2, 1, b_neighbors, directed)
		for n in b_neighbors:
			result += snap.GetCmnNbrs(G, n, n1)
	if deleted: G.AddEdge(n1, n2)
	return result

#returns the length of the shortest path between 2 nodes
def get_graph_distance(G, n1, n2, directed=False):
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True
	result = -1*snap.GetShortPath(G, n1, n2, directed)
	if deleted: G.AddEdge(n1, n2)
	return result

#returns the number of common neighbors between two nodes
def get_common_neighbors(G, n1, n2):
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True
	result = snap.GetCmnNbrs(G, n1, n2)
	if deleted: G.AddEdge(n1, n2)
	return result

#returns the jaccard coefficient between two nodes
#assumes the graph is undirected
def jaccard_coefficient(G, n1, n2):
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True
	common_neighbors = snap.GetCmnNbrs(G, n1, n2)
	total_neighbors = G.GetNI(n1).GetDeg() + G.GetNI(n2).GetDeg()
	result = 0.0 if total_neighbors == 0 else float(common_neighbors)/total_neighbors
	if deleted: G.AddEdge(n1, n2)
	return result

#returns the adamic adar score between two nodes
def adamic_adar(G, n1, n2, directed = False):
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True
	n1_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n1, 1, n1_neighbors, directed)
	n2_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n2, 1, n2_neighbors, directed)
	total_neighbors = set(n1_neighbors) & set(n2_neighbors)

	aa = 0.0
	for n in total_neighbors:
		aa += 1.0/math.log(G.GetNI(n).GetDeg())

	if deleted: G.AddEdge(n1, n2)
	return aa

#returns the preferential attachment score between two nodes
def preferential_attachment(G, n1, n2):
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True
	nodel1 = G.GetNI(n1)
	nodel2 = G.GetNI(n2)
	result = nodel1.GetOutDeg() * nodel2.GetOutDeg()
	if deleted: G.AddEdge(n1, n2)
	return result

#returns the katz score between two nodes
def katz_measure(G, n1, n2, beta=0.05, directed = False):
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True

	result = 0
	for i in range(1,6):
		visited = set([n1])
		result += beta**i*get_num_paths(G, i, n1, n2, visited, directed)
		print result
	'''
	result = 0
	if G.IsEdge(n1, n2): result += beta
	n = [n1]
	for i in range(2, 6):
		print i
		new_n = set()
		for ni in n:
			result += beta**i*snap.GetCmnNbrs(G, ni, n2)
			n1_neighbors = snap.TIntV()
			snap.GetNodesAtHop(G, ni, 1, n1_neighbors, directed)
			new_n.update(n1_neighbors)
		n = list()
	'''
	if deleted: G.AddEdge(n1, n2)
	return result

def get_num_paths(G, i, n1, n2, visited, directed):
	if i == 0:
		if n1 == n2: return 1
		return 0
	n1_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n1, 1, n1_neighbors, directed)
	result = 0
	for n in n1_neighbors:
		if n not in visited:
			visited.add(n)
			result += get_num_paths(G, i-1, n, n2, visited, directed)
			visited.remove(n)
	return result

		

	'''
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True
	index = 0
	node_map = {}
	for n in G.Nodes():
		node_map[n.GetID()] = index
		index += 1
	idenitity_matrix = np.identity(G.GetNodes())
	adjacency_matrix = get_adjacency_matrix(G, node_map)
	katz_scores = numpy.linalg.inv(idenitity_matrix-beta * adjacency_matrix) - idenitity_matrix
	if deleted: G.AddEdge(n1, n2)
	return katz_scores[n1][n2]
	'''

#generates the adjacency matrix for the graph
def get_adjacency_matrix(G, node_map):
	'''
	n_rows = G.GetNodes()
	A = np.zeros(shape =(n_rows, n_rows))
	for edge in G.Edges():
		A[edge.GetSrcNId()][edge.GetDstNId()]  = 1
		A[edge.GetDstNId()][edge.GetSrcNId()]  = 1
	return A
	'''
	A = scipy.sparse.csr_matrix((G.GetNodes(), G.GetNodes()), dtype=np.int8).toarray()
	for edge in G.Edges():
		A[node_map[edge.GetSrcNId()]][node_map[edge.GetDstNId()]]  = 1
		A[node_map[edge.GetDstNId()]][node_map[edge.GetSrcNId()]]  = 1
	return A

def sim_rank(G, n1, n2, gamma=0.8, directed=False):
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True
	sim_rank_wrapper(G, n1, n2, gamma)
	if deleted: G.AddEdge(n1, n2)

def sim_rank_wrapper(G, n1, n2, gamma, directed=False):
	if n1 == n2: return 1
	constant = gamma/preferential_attachment(G, n1, n2)
	n1_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n1, 1, n1_neighbors, directed)
	n2_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n2, 1, n2_neighbors, directed)
	result = 0
	for a in n1_neighbors:
		for b in n2_neighbors:
			result += sim_rank_wrapper(G, a, b, gamma)
	return result * constant

#returns the expected number of steps required for a random walk starting
#at n1 to reach n2
def hitting_time(G, n1, n2, num_steps = 1000):
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True

	adjacency_matrix = get_adjacency_matrix(G)

	# Make the final state an absorbing condition
	adjacency_matrix[n2,:] = 0
	adjacency_matrix[n2,n2] = 1

	# Make a proper Markov matrix by row normalizing
	adjacency_matrix = (adjacency_matrix.T/adjacency_matrix.sum(axis=1)).T

	B = adjacency_matrix.copy()
	for n in xrange(num_steps):
		B = dot(B,A)
	if deleted: G.AddEdge(n1, n2)
	return -B[n1][n2]

#returns the expected number of steps for a random walk between n1 to reach n2
def commute_time(G, n1, n2, num_steps=1000):
	return hitting_time(G, n1, n2, num_steps) + hitting_time(G, n2, n1, num_steps)

#returns the normalized expected number of steps required for a random walk starting
#at n1 to reach n2
def hitting_time_normalized(G, n1, n2, num_steps = 1000):
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True

	adjacency_matrix = get_adjacency_matrix(G)

	# Make the final state an absorbing condition
	adjacency_matrix[n2,:] = 0
	adjacency_matrix[n2,n2] = 1

	# Make a proper Markov matrix by row normalizing
	adjacency_matrix = (adjacency_matrix.T/adjacency_matrix.sum(axis=1)).T

	B = adjacency_matrix.copy()
	for n in xrange(num_steps):
		B = dot(B,A)
	if deleted: G.AddEdge(n1, n2)
	return -B[n1][n2]/float(sum(B[:,n2]))

#returns the normalized expected number of steps for a random walk between n1 to reach n2
def communte_time_normalized(G, n1, n2, num_steps = 1000):
	return hitting_time(G, n1, n2, stationary_probability_n2, num_steps) + hitting_time(G, n2, n1, stationary_probability_n1, num_steps)

#returns the pagerank between n1 to reach n2
def rooted_page_rank(G, n1, n2, num_steps=1000, alpha = 0.2):
	return hitting_time(G, n1, n2, num_steps) * alpha

#returns if nodes n1 and n2 are in the same community
def same_community(G, n1, n2, method = "CNM"):
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True

	CmtyV = snap.TCnComV()
	if method == "CNM":
		modularity = snap.CommunityCNM(G, CmtyV)
	elif method == "GN":
		modularity = snap.CommunityGirvanNewman(G, CmtyV)

	for cmty in CmtyV:
		cmty_set = set(cmty)
		if n1 in cmty and n2 in cmty:
			if deleted: G.AddEdge(n1, n2)
			return 1
	if deleted: G.AddEdge(n1, n2)
	return 0

#returns the number of connections between n1 and n2 neighborhoods
def friends_measure(G, n1, n2, directed=False):
	deleted = False
	if G.IsEdge(n1, n2):
		G.DelEdge(n1, n2)
		deleted = True

	n1_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n1, 1, n1_neighbors, directed)
	n2_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n2, 1, n2_neighbors, directed)
	fm = 0
	for a in n1_neighbors:
		for b in n2_neighbors:
			if a == b or G.IsEdge(a, b) or G.IsEdge(b,a):
				fm += sim_rank(G, a, b)
	if deleted: G.AddEdge(n1, n2)
	return fm


#################################
######NODE DEGREE MEASURES#######
#################################

def get_in_degree(G, n):
	# InDegV = snap.TIntPrV()
	# snap.GetNodeInDegV(Graph, InDegV)
	# for item in InDegV:
	# 	if item.GetVal1() == n: return item.GetVal2()
	return G.GetNI(n).GetInDeg()

def get_out_degree(G, n):
	return G.GetNI(n).GetOutDeg()


#################################
####NODE CENTRALITY MEASURES#####
#################################
def get_degree_centrality(G, n):
	return snap.GetDegreeCentr(G, n)

def get_node_betweenness_centrality(G, n, directed = False):
	nodes = snap.TIntFltH()
	edges = snap.TIntPrFltH()
	snap.GetBetweennessCentr(G, nodes, edges, 1.0)
	return nodes[n]

def get_closeness_centrality(G, n, directed = False):
	return snap.GetClosenessCentr(G, n, True, directed)

#returns the average shortest path length to all other nodes that reside in the
#connected component of the given node
def get_farness_centrality(G, n, directed = False):
	return snap.GetFarnessCentr(G, n, IsDir=directed)

def get_page_rank(G, n):
	PRankH = snap.TIntFltH()
	snap.GetPageRank(G, PRankH)
	return PRankH[n]

#returns the HITS score of a given node as a hub score, authority score tuple 
def get_HITS_scores(G, n):
	NIdHubH = snap.TIntFltH()
	NIdAuthH = snap.TIntFltH()
	snap.GetHits(G, NIdHubH, NIdAuthH)
	return NIdHubH[n], NIdAuthH[n]

#returns the largest shortest-path distance from a given node n
#to any other node in the graph G
def get_node_eccentricity(G, n, directed = False):
	return snap.GetNodeEcc(G, n, directed)

def get_edge_betweenness_centrality(G, e, directed = False):
	nodes = snap.TIntFltH()
	edges = snap.TIntPrFltH()
	snap.GetBetweennessCentr(G, nodes, edges, 1.0)
	return edges[(e.GetVal1()), e.GetVal2()]

#################################
######PATH LENGTH MEASURES#######
#################################

#returns the length of the shortest path between two given nodes
def get_shortest_path_to_one_node(G, src_n, dst_n, directed=False):
	return GetShortPath(G, src_n, dst_n, directed)

#returns the length of the shortest path from a given node and 
#a mapping of node id to path length for the shortest path
#from a given node to each node in the mapping
def get_shortest_path_to_all_nodes(G, src_n, directed=False):
	NIdToDistH = snap.TIntH()
	shortestPath = snap.GetShortPath(G, src_n, NIdToDistH, directed)
	return shortestPath, NIdToDistH

#returns the diameter of a graph or subgraph
#the calculation is approximate and uses num_start_nodes as
#the number of randomly chosen starting nodes to use for the calculation
def get_longest_shortest_path(G, num_start_nodes, directed=False):
	return snap.GetBfsFullDiam(G, num_start_nodes, directed)

####################################
######CONNECTED COMP MEASURES#######
####################################

#returns if a graph or subgraph is connected
def is_graph_connected(G):
	return snap.IsConnected(G)

#returns if a graph or subgraph is weakly connected
def is_graph_weakly_connected(G):
	return snap.IsWeaklyConn(G)

#returns the size of the connected component in which a node lies
def get_size_of_conn_comp(G, n):
	CnCom = snap.TIntV()
	return len(snap.GetNodeWcc(G, n, CnCom))

#returns true if a node is an articulation point and false otherwise
def is_articulation_point(G, n):
	ArtNIdV = snap.TIntV()
	snap.GetArtPoints(G, ArtNIdV)
	return n in set(ArtNIdV)

#returns true if an edge is a bridge and false otherwise
def is_edge_a_bridge(G, e):
	EdgeV = snap.TIntPrV()
	snap.GetEdgeBridges(G, EdgeV)
	return (e.GetVal1(), e.GetVal2()) in set(EdgeV)




####################################
########CLUSTERING MEASURES#########
####################################

def get_modularity(G, nodes):
	return snap.GetModularity(G, nodes, G.GetEdges())

def get_number_of_shared_neighbors(G, n1, n2):
	return snap.GetCmnNbrs(G, n1, n2)

def get_clustering_coefficient(G, n):
	return snap.GetNodeClustCf(G, n)

def get_number_of_triads_with_node(G, n):
	return snap.GetNodeTriads(G, n)

#returns a vector of the number of nodes reachable from node n in hops less than
#max_hops number of hops
def get_approximate_neighborhood(G, n, max_hops, directed=False, approx=32):
	DistNbrsV = snap.TIntFltKdV()
	snap.GetAnf(G, n, DistNbrsV, max_hops, directed, approx)


#############################
# New Feature Functions (AVP)
#############################

def get_betw_sum(G, n1, n2):
	return get_node_betweenness_centrality(G, n1) + \
			get_node_betweenness_centrality(G, n2)

def get_degree_sum(G, n1, n2):
	return get_degree_centrality(G, n1) + \
			get_degree_centrality(G, n2)

def get_closeness_sum(G, n1, n2):
	return get_closeness_centrality(G, n1) + \
			get_closeness_centrality(G, n2)

def get_coeff_sum(G, n1, n2):
	return get_clustering_coefficient(G, n1) + \
			get_clustering_coefficient(G, n2)

def get_page_rank_sum(G, n1, n2):
	return get_page_rank(G, n1) + get_page_rank(G, n2)

def get_hubs_sum(G, n1, n2):
	return get_HITS_scores(G, n1)[0] + get_HITS_scores(G, n2)[0]

def get_auths_sum(G, n1, n2):
	return get_HITS_scores(G, n1)[1] + get_HITS_scores(G, n2)[1]
