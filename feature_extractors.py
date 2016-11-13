import snap
import math

#####################################
######LINK PREDICTION MEASURES#######
#####################################

#returns the length of the shortest path between 2 nodes
def get_graph_distance(G, n1, n2, directed=False):
	return -GetShortPath(G, n1, n2, directed):

#returns the number of common neighbors between two nodes
def get_common_neighbors(G, n1, n2):
	return snap.GetCmnNbrs(G, n1, n2)

#returns the jaccard coefficient between two nodes
#assumes the graph is undirected
def jaccard_coefficient(G, n1, n2):
	common_neighbors = snap.GetCmnNbrs(G, n1, n2)
	total_neighbors = get_out_degree(G, n1) + get_out_degree(G, n2)
	return 0.0 if total_neighbors == 0 else float(common_neighbors)/total_neighbors

#returns the adamic adar score between two nodes
def adamic_adar(G, n1, n2, directed = False):
	n1_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n1, 1, n1_neighbors, directed)
	n2_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n2, 1, n2_neighbors, directed)
	total_neighbors = set(n1_neighbors) & set(n2_neighbors)

	aa = 0.0
	for n in total_neighbors:
		aa += 1.0/math.log(get_out_degree(G, n))

	return aa

#returns the preferential attachment score between two nodes
def preferential_attachment(G, n1, n2):
	return get_out_degree(G, n1) * get_out_degree(G, n2)

#returns the katz score between two nodes
def katz_measure(G, n1, n2, beta):
	adjacency_matrix = get_adjacency_matrix(G)
	idenitity_matrix = np.identity(G.GetNodes())
	katz_scores = numpy.linalg.inv(idenitity_matrix-beta * adjacency_matrix) - idenitity_matrix
	return katz_scores[n1][n2]

#generates the adjacency matrix for the graph
def get_adjacency_matrix(G):
	n_rows = G.GetNodes()
	A = np.zeros(shape =(n_rows, n_rows))
	for edge in G.Edges():
		A[edge.GetSrcNId()][edge.GetDstNId()]  = 1
		A[edge.GetDstNId()][edge.GetSrcNId()]  = 1

def sim_rank(G, n1, n2, gamma):
	if n1 == n2: return 1

	constant = gamma/preferential_attachment(G, n1, n2)
	n1_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n1, 1, n1_neighbors, directed)
	n2_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n2, 1, n2_neighbors, directed)
	result = 0
	for a in n1_neighbors:
		for b in n2_neighbors:
			result += sim_rank(G, a, b)
	return result * constant

#returns the expected number of steps required for a random walk starting
#at n1 to reach n2
def hitting_time(G, n1, n2, num_steps):
	adjacency_matrix = get_adjacency_matrix(G)

	# Make the final state an absorbing condition
	adjacency_matrix[n2,:] = 0
	adjacency_matrix[n2,n2] = 1

	# Make a proper Markov matrix by row normalizing
	adjacency_matrix = (adjacency_matrix.T/adjacency_matrix.sum(axis=1)).T

	B = adjacency_matrix.copy()
	for n in xrange(num_steps):
    	B = dot(B,A)
    return -B[n1][n2]

#returns the expected number of steps for a random walk between n1 to reach n2
def communte_time(G, n1, n2, num_steps):
	return hitting_time(G, n1, n2, num_steps) + hitting_time(G, n2, n1, num_steps)

#returns the normalized expected number of steps required for a random walk starting
#at n1 to reach n2
def hitting_time_normalized(G, n1, n2, stationary_probability_n2, num_steps):
	adjacency_matrix = get_adjacency_matrix(G)

	# Make the final state an absorbing condition
	adjacency_matrix[n2,:] = 0
	adjacency_matrix[n2,n2] = 1

	# Make a proper Markov matrix by row normalizing
	adjacency_matrix = (adjacency_matrix.T/adjacency_matrix.sum(axis=1)).T

	B = adjacency_matrix.copy()
	for n in xrange(num_steps):
    	B = dot(B,A)
    return -B[n1][n2]*stationary_probability_n2

#returns the normalized expected number of steps for a random walk between n1 to reach n2
def communte_time_normalized(G, n1, n2, stationary_probability_n1, stationary_probability_n2, num_steps):
	return hitting_time(G, n1, n2, stationary_probability_n2, num_steps) + hitting_time(G, n2, n1, stationary_probability_n1, num_steps)

#returns the pagerank between n1 to reach n2
def rooted_page_rank(G, n1, n2, num_steps, alpha):
	return hitting_time(G, n1, n2, num_steps) * alpha

#returns if nodes n1 and n2 are in the same community
def same_community(G, n1, n2, method):
	CmtyV = snap.TCnComV()
	if method == "CNM":
		modularity = snap.CommunityCNM(UGraph, CmtyV)
	elif method == "GN"
		modularity = snap.CommunityGirvanNewman(UGraph, CmtyV)

	for cmty in CmtyV:
		cmty_set = set(cmty)
		if n1 in cmty and n2 in cmty: return 1

	return 0

#returns the number of connections between n1 and n2 neighborhoods
def friends_measure():
	n1_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n1, 1, n1_neighbors, directed)
	n2_neighbors = snap.TIntV()
	snap.GetNodesAtHop(G, n2, 1, n2_neighbors, directed)
	fm = 0
	for a in n1_neighbors:
		for b in n2_neighbors:
			if a == b or G.IsEdge(a, b) or G.IsEdge(b,a):
				fm += sim_rank(G, a, b)
	return fm


#################################
######NODE DEGREE MEASURES#######
#################################

def get_in_degree(G, n):
	InDegV = snap.TIntPrV()
	snap.GetNodeInDegV(Graph, InDegV)
	for item in InDegV:
    	if item.GetVal1() == n.GetId(): return item.GetVal2()

def get_out_degree(G, n):
	InDegV = snap.TIntPrV()
	snap.GetNodeOutDegV(Graph, InDegV)
	for item in InDegV:
    	if item.GetVal1() == n.GetId(): return item.GetVal2()

def get_total_degree(G, n):

#################################
####NODE CENTRALITY MEASURES#####
#################################

def get_degree_centrality(G, n):
	return snap.GetDegreeCentr(G, n.GetId())

def get_node_betweenness_centrality(G, n, directed = False):
	nodes = snap.TIntFltH()
	edges = snap.TIntPrFltH()
	snap.GetBetweennessCentr(G, nodes, edges, 1.0)
	return nodes[n.GetId()]

def get_closeness_centrality(G, n, directed = False):
	return snap.GetClosenessCentr(G, n.GetId(), IsDir=directed)

#returns the average shortest path length to all other nodes that reside in the
#connected component of the given node
def get_farness_centrality(G, n, directed = False):
	return snap.GetFarnessCentr(G, n.GetId(), IsDir=directed)

def get_page_rank(G, n):
	PRankH = snap.TIntFltH()
	snap.GetPageRank(G, PRankH)
	return PRankH[n.GetId()]

#returns the HITS score of a given node as a hub score, authority score tuple 
def get_HITS_scores(G, n):
	NIdHubH = snap.TIntFltH()
	NIdAuthH = snap.TIntFltH()
	snap.GetHits(Graph, NIdHubH, NIdAuthH)
	return NIdHubH[n.GetId(), NIdAuthH[n.GetId()]

#returns the largest shortest-path distance from a given node n
#to any other node in the graph G
def get_node_eccentricity(G, n, directed = False):
	return snap.GetNodeEcc(G, n.GetId(), directed)

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
	return GetShortPath(G, src_n.GetId(), dst_n.GetId(), directed)

#returns the length of the shortest path from a given node and 
#a mapping of node id to path length for the shortest path
#from a given node to each node in the mapping
def get_shortest_path_to_all_nodes(G, src_n, directed=False):
	NIdToDistH = snap.TIntH()
	shortestPath = snap.GetShortPath(G, src_n.GetId(), NIdToDistH, directed)
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
	return len(snap.GetNodeWcc(G, n.GetId(), CnCom))

#returns true if a node is an articulation point and false otherwise
def is_articulation_point(G, n):
	ArtNIdV = snap.TIntV()
	snap.GetArtPoints(G, ArtNIdV)
	return n.GetId() in set(ArtNIdV)

#returns true if an edge is a bridge and false otherwise
def is_edge_a_bridge(G, e):
	EdgeV = snap.TIntPrV()
	snap.GetEdgeBridges(UGraph, EdgeV)
	return (e.GetVal1(), e.GetVal2()) in set(EdgeV)




####################################
########CLUSTERING MEASURES#########
####################################

def get_modularity(G, nodes):
	return snap.GetModularity(G, nodes, G.GetEdges())

def get_number_of_shared_neighbors(G, n1, n2):
	return snap.GetCmnNbrs(G, n1.GetId(), n2.GetId())

def get_cluseting_coefficient(G, n):
	return snap.GetNodeClustCf(G, n.GetId())

def get_number_of_triads_with_node(G, n):
	return snap.GetNodeTriads(G, n.GetId())

#returns a vector of the number of nodes reachable from node n in hops less than
#max_hops number of hops
def get_approximate_neighborhood(G, n, max_hops, directed=false, approx=32):
	DistNbrsV = snap.TIntFltKdV()
	snap.GetAnf(G, n.GetId(), DistNbrsV, max_hops, directed, approx)