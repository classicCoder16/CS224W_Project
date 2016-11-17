'''
This file provides a function that takes a graph, and outputs a file consisting
of pairs of node ids with labels. Each label represents
whether an edge exists between these node ids. 
'''

import random
import sys
import numpy as np


'''
Function: get_n_edges
---------------------
Function that, given a graph and a number,
returns n unique pairs of either connected nodes 
or unconnected nodes, depending on the value of pos. 
'''
def get_n_test_edges(train_graph, test_graph, train_ex, n, pos=True):
	curr_edges = set()

	# Keep randomly sampling pairs of nodes
	# until we get n valid, unique pairs.

	all_train_ex = set(train_ex)
	while len(curr_edges) != n:

		# Randomly generate ids
		src_id = train_graph.GetRndNId()
		dst_id = train_graph.GetRndNId()

		# No self-loops
		if src_id == dst_id: continue

		# Form the candidate pair
		cand_pair = tuple(sorted([src_id, dst_id]))

		if cand_pair in all_train_ex: continue

		# Else, if we want negative samples, and it is not an edge,
		# add the pair to the set. 
		if not pos and not train_graph.IsEdge(cand_pair[0], cand_pair[1]) \
		and not test_graph.IsEdge(cand_pair[0], cand_pair[1]):
			curr_edges.add(cand_pair)

	# Return the set of pairs as a list.
	return list(curr_edges)


'''
Function: get_n_edges
---------------------
Function that, given a graph and a number,
returns n unique pairs of either connected nodes 
or unconnected nodes, depending on the value of pos. 
'''
def get_n_edges(graph, n, pos=True):
	curr_edges = set()

	# Keep randomly sampling pairs of nodes
	# until we get n valid, unique pairs.
	while len(curr_edges) != n:

		# Randomly generate ids
		src_id = graph.GetRndNId()
		dst_id = graph.GetRndNId()

		# No self-loops
		if src_id == dst_id: continue

		# Form the candidate pair
		cand_pair = tuple(sorted([src_id, dst_id]))

		# If we want positive samples, and it is an existing edge,
		# add the pair to the set
		if pos and graph.IsEdge(cand_pair[0], cand_pair[1]):
			curr_edges.add(cand_pair)

		# Else, if we want negative samples, and it is not an edge,
		# add the pair to the set. 
		elif not pos and not graph.IsEdge(cand_pair[0], cand_pair[1]):
			curr_edges.add(cand_pair)

	# Return the set of pairs as a list.
	return list(curr_edges)

def get_pos_test_edges(train_graph, test_graph, train_ex, num_pos):
	all_edges = []
	all_train_ex = set(train_ex)
	for edge in test_graph.Edges():
		cand_pair = tuple(sorted([edge.GetSrcNId(), edge.GetDstNId()]))
		if cand_pair in all_train_ex: continue
		src_id = cand_pair[0]
		dst_id = cand_pair[1]
		if not train_graph.IsNode(src_id) or not train_graph.IsNode(dst_id): continue
		all_edges.append(cand_pair)
	return random.sample(all_edges, num_pos)


'''
Function: extract_examples
--------------------------
Function that returns num pos + num neg
id/label pairs from the given graph
'''
def extract_examples(graph, num_pos, num_neg):
	print 'Extracting examples from graph...'
	# Get pos edges and labels
	print 'Getting positve edges...'
	# pos_edges = get_n_edges(graph, num_pos)
	pos_edges = random.sample([tuple(sorted((edge.GetSrcNId(), edge.GetDstNId()))) \
					for edge in graph.Edges()], num_pos)
	pos_labels = [1]*len(pos_edges)

	# Get negative edges and labels
	print 'Getting negative edges...'
	neg_edges = get_n_edges(graph, num_neg, pos=False)
	neg_labels = [-1]*len(neg_edges)

	# Concatenate together
	all_pairs = pos_edges + neg_edges 
	all_labels = pos_labels + neg_labels
	print 'Done!'
	return all_pairs, all_labels 

'''
Function: extract_examples
--------------------------
Function that returns num pos + num neg
id/label pairs from the given graph
'''
def extract_test_examples(train_graph, test_graph, train_ex, num_pos, num_neg):
	print 'Extracting examples from graph...'
	# Get pos edges and labels
	print 'Getting positve edges...'
	pos_edges = get_pos_test_edges(train_graph, test_graph, train_ex, num_pos)
	pos_labels = [1]*len(pos_edges)

	# Get negative edges and labels
	print 'Getting negative edges...'
	neg_edges = get_n_test_edges(train_graph, test_graph, train_ex, num_neg, pos=False)
	neg_labels = [-1]*len(neg_edges)

	# Concatenate together
	all_pairs = pos_edges + neg_edges 
	all_labels = pos_labels + neg_labels
	print 'Done!'
	return all_pairs, all_labels 


'''
Function: write_examples
--------------------------
Function that writes the node pairs and their
labels to a file.
'''
def write_examples(filename, node_pairs, labels):
	print 'Saving node pairs and labels to file...'
	f = open(filename, 'w')
	# Write every line as tab-separated
	for i in range(len(nodes)):
		line = str(node_pairs[i][0]) + '\t' + str(node_pairs[i][1]) + '\t' + str(node_pairs[i]) + '\n'
		f.write(line)
	f.close()
	print 'Done!'

