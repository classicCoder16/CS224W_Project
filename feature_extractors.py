import snap

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