'''
This file provides a function that takes a graph, and outputs a file consisting
of pairs of node ids with labels. Each label represents
whether an edge exists between these node ids. 
'''

import random
import sys
import numpy as np


def is_pin_id(nid, board_ids):
	return nid > board_ids[1]

def is_board_id(nid, board_ids):
	return nid >= board_ids[0] and nid <= board_ids[1]

def is_user_id(nid, board_ids):
	return nid < board_ids[0]

'''
Function: get_n_edges
---------------------
Function that, given a graph and a number,
returns n unique pairs of either connected nodes 
or unconnected nodes, depending on the value of pos. 
'''
def get_n_test_edges(train_graph, test_graph, train_ex, n, board_ids):
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
		if board_ids is not None:
			if (isBoard(src_id, board_ids) and isBoard(dst_id, board_ids)) or \
				((not isBoard(src_id, board_ids)) and (not isBoard(dst_id, board_ids))): continue

		# Form the candidate pair
		cand_pair = tuple(sorted([src_id, dst_id]))

		if cand_pair in all_train_ex: continue

		# Else, if we want negative samples, and it is not an edge,
		# add the pair to the set. 
		if not train_graph.IsEdge(cand_pair[0], cand_pair[1]) \
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
def get_n_edges(graph, n):
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

		# Else, if we want negative samples, and it is not an edge,
		# add the pair to the set. 
		if not graph.IsEdge(cand_pair[0], cand_pair[1]):
			curr_edges.add(cand_pair)

	# Return the set of pairs as a list.
	return list(curr_edges)


def isBoard(n, board_ids):
	return n >= board_ids[0] and n <= board_ids[1]

def get_pos_test_edges(train_graph, test_graph, train_ex, num_pos, board_ids):
	all_edges = []
	all_train_ex = set(train_ex)
	for edge in test_graph.Edges():
		cand_pair = tuple(sorted([edge.GetSrcNId(), edge.GetDstNId()]))
		if cand_pair in all_train_ex: continue
		src_id = cand_pair[0]
		dst_id = cand_pair[1]
		if train_graph.IsEdge(src_id, dst_id):
			print 'HERE'
			continue
		if not train_graph.IsNode(src_id) or not train_graph.IsNode(dst_id): continue
		if board_ids is not None:
			if (isBoard(src_id, board_ids) and isBoard(dst_id, board_ids)) or \
				((not isBoard(src_id, board_ids)) and (not isBoard(dst_id, board_ids))): continue
		all_edges.append(cand_pair)
	return random.sample(all_edges, num_pos)

def get_neg_foll_edges(graph, num_neg, board_ids):
	curr_edges = set()

	# Keep randomly sampling pairs of nodes
	# until we get n valid, unique pairs.
	while len(curr_edges) != num_neg:

		# Randomly generate ids
		src_id = graph.GetRndNId()
		dst_id = graph.GetRndNId()

		# No self-loops
		if src_id == dst_id: continue
		if (is_user_id(src_id, board_ids) and is_board_id(dst_id, board_ids)) or \
			(is_user_id(dst_id, board_ids) and is_board_id(src_id, board_ids)):
			
			# Form the candidate pair
			cand_pair = tuple(sorted([src_id, dst_id]))

			# Else, if we want negative samples, and it is not an edge,
			# add the pair to the set. 
			if not graph.IsEdge(cand_pair[0], cand_pair[1]):
				curr_edges.add(cand_pair)

	# Return the set of pairs as a list.
	return list(curr_edges)

def get_neg_pin_edges(graph, num_neg, board_ids):
	curr_edges = set()

	# Keep randomly sampling pairs of nodes
	# until we get n valid, unique pairs.
	while len(curr_edges) != num_neg:

		# Randomly generate ids
		src_id = graph.GetRndNId()
		dst_id = graph.GetRndNId()

		# No self-loops
		if src_id == dst_id: continue
		if (is_pin_id(src_id, board_ids) and is_board_id(dst_id, board_ids)) or \
			(is_pin_id(dst_id, board_ids) and is_board_id(src_id, board_ids)):
			
			# Form the candidate pair
			cand_pair = tuple(sorted([src_id, dst_id]))

			# Else, if we want negative samples, and it is not an edge,
			# add the pair to the set. 
			if not graph.IsEdge(cand_pair[0], cand_pair[1]):
				curr_edges.add(cand_pair)

	# Return the set of pairs as a list.
	return list(curr_edges)

def get_pos_test_pin_edges(train_graph, test_graph, train_ex, num_pos, board_ids):
	all_edges = []
	all_train_ex = set(train_ex)
	for edge in test_graph.Edges():
		cand_pair = tuple(sorted([edge.GetSrcNId(), edge.GetDstNId()]))
		if cand_pair in all_train_ex: continue
		src_id = cand_pair[0]
		dst_id = cand_pair[1]
		if not train_graph.IsNode(src_id) or not train_graph.IsNode(dst_id): continue
		if (is_pin_id(src_id, board_ids) and is_board_id(dst_id, board_ids)) or \
			(is_pin_id(dst_id, board_ids) and is_board_id(src_id, board_ids)):
				all_edges.append(cand_pair)
	return random.sample(all_edges, num_pos)	

def get_neg_test_pin_edges(train_graph, test_graph, train_ex, num_neg, board_ids):
	curr_edges = set()

	# Keep randomly sampling pairs of nodes
	# until we get n valid, unique pairs.

	all_train_ex = set(train_ex)
	while len(curr_edges) != num_neg:

		# Randomly generate ids
		src_id = train_graph.GetRndNId()
		dst_id = train_graph.GetRndNId()

		# No self-loops
		if src_id == dst_id: continue
		if (is_pin_id(src_id, board_ids) and is_board_id(dst_id, board_ids)) or \
			(is_pin_id(dst_id, board_ids) and is_board_id(src_id, board_ids)):

			# Form the candidate pair
			cand_pair = tuple(sorted([src_id, dst_id]))

			if cand_pair in all_train_ex: continue

			# Else, if we want negative samples, and it is not an edge,
			# add the pair to the set. 
			if not train_graph.IsEdge(cand_pair[0], cand_pair[1]) \
			and not test_graph.IsEdge(cand_pair[0], cand_pair[1]):
				curr_edges.add(cand_pair)

	# Return the set of pairs as a list.
	return list(curr_edges)

def get_pos_test_fol_edges(train_graph, test_graph, train_ex, num_pos, board_ids):
	all_edges = []
	all_train_ex = set(train_ex)
	for edge in test_graph.Edges():
		cand_pair = tuple(sorted([edge.GetSrcNId(), edge.GetDstNId()]))
		if cand_pair in all_train_ex: continue
		src_id = cand_pair[0]
		dst_id = cand_pair[1]
		if not train_graph.IsNode(src_id) or not train_graph.IsNode(dst_id): continue
		if (is_user_id(src_id, board_ids) and is_board_id(dst_id, board_ids)) or \
			(is_user_id(dst_id, board_ids) and is_board_id(src_id, board_ids)):
				all_edges.append(cand_pair)
	return random.sample(all_edges, num_pos)	

def get_neg_test_fol_edges(train_graph, test_graph, train_ex, num_neg, board_ids):
	curr_edges = set()

	# Keep randomly sampling pairs of nodes
	# until we get n valid, unique pairs.

	all_train_ex = set(train_ex)
	while len(curr_edges) != num_neg:

		# Randomly generate ids
		src_id = train_graph.GetRndNId()
		dst_id = train_graph.GetRndNId()

		# No self-loops
		if src_id == dst_id: continue
		if (is_user_id(src_id, board_ids) and is_board_id(dst_id, board_ids)) or \
			(is_user_id(dst_id, board_ids) and is_board_id(src_id, board_ids)):

			# Form the candidate pair
			cand_pair = tuple(sorted([src_id, dst_id]))

			if cand_pair in all_train_ex: continue

			# Else, if we want negative samples, and it is not an edge,
			# add the pair to the set. 
			if not train_graph.IsEdge(cand_pair[0], cand_pair[1]) \
			and not test_graph.IsEdge(cand_pair[0], cand_pair[1]):
				curr_edges.add(cand_pair)

	# Return the set of pairs as a list.
	return list(curr_edges)


def get_pin_tr_ex(graph, num_pos, num_neg, board_ids):
	all_edges = [tuple(sorted((edge.GetSrcNId(), edge.GetDstNId()))) \
					for edge in graph.Edges()]
	all_pin_edges = []
	for edge in all_edges:
		src_id = edge[0]
		dst_id = edge[1]
		if (is_pin_id(src_id, board_ids) and is_board_id(dst_id, board_ids)) or \
			(is_pin_id(dst_id, board_ids) and is_board_id(src_id, board_ids)):
			all_pin_edges.append(edge)
	pos_edges = random.sample(all_pin_edges, num_pos)
	pos_labels = [1]*len(pos_edges)
	neg_edges = get_neg_pin_edges(graph, num_neg, board_ids)
	neg_labels = [-1]*len(neg_edges)

	# Concatenate together
	all_pairs = pos_edges + neg_edges 
	all_labels = pos_labels + neg_labels
	print 'Done!'
	return all_pairs, all_labels 




def get_foll_tr_ex(graph, num_pos, num_neg, board_ids, attributes):
	all_edges = [tuple(sorted((edge.GetSrcNId(), edge.GetDstNId()))) \
					for edge in graph.Edges()]
	all_foll_edges = []
	for edge in all_edges:
		src_id = edge[0]
		dst_id = edge[1]
		if (is_user_id(src_id, board_ids) and is_board_id(dst_id, board_ids)) or \
			(is_user_id(dst_id, board_ids) and is_board_id(src_id, board_ids)):
			if 'create_time' in attributes[tuple(sorted([src_id, dst_id], reverse=True))]: continue
			all_foll_edges.append(edge)
	pos_edges = random.sample(all_foll_edges, num_pos)
	pos_labels = [1]*len(pos_edges)
	neg_edges = get_neg_foll_edges(graph, num_neg, board_ids)
	neg_labels = [-1]*len(neg_edges)

	# Concatenate together
	all_pairs = pos_edges + neg_edges 
	all_labels = pos_labels + neg_labels
	print 'Done!'
	return all_pairs, all_labels 


def get_pin_tst_ex(train_graph, test_graph, train_ex, num_pos, num_neg, board_ids):
	print 'Extracting pinned test examples from graph...'
	# Get pos edges and labels
	print 'Getting positve edges...'
	pos_edges = get_pos_test_pin_edges(train_graph, test_graph, train_ex, num_pos, board_ids)
	pos_labels = [1]*len(pos_edges)

	# Get negative edges and labels
	print 'Getting negative edges...'
	neg_edges = get_neg_test_pin_edges(train_graph, test_graph, train_ex, num_neg, board_ids)
	neg_labels = [-1]*len(neg_edges)

	# Concatenate together
	all_pairs = pos_edges + neg_edges 
	all_labels = pos_labels + neg_labels
	print 'Done!'
	return all_pairs, all_labels 

def get_foll_tst_ex(train_graph, test_graph, train_ex, num_pos, num_neg, board_ids):
	print 'Extracting foll test examples from graph...'
	# Get pos edges and labels
	print 'Getting positve edges...'
	pos_edges = get_pos_test_fol_edges(train_graph, test_graph, train_ex, num_pos, board_ids)
	pos_labels = [1]*len(pos_edges)

	# Get negative edges and labels
	print 'Getting negative edges...'
	neg_edges = get_neg_test_fol_edges(train_graph, test_graph, train_ex, num_neg, board_ids)
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
	neg_edges = get_n_edges(graph, num_neg)
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
def extract_test_examples(train_graph, test_graph, train_ex, num_pos, num_neg, board_ids=None):
	print 'Extracting examples from graph...'
	# Get pos edges and labels
	print 'Getting positve edges...'
	pos_edges = get_pos_test_edges(train_graph, test_graph, train_ex, num_pos, board_ids)
	pos_labels = [1]*len(pos_edges)

	# Get negative edges and labels
	print 'Getting negative edges...'
	neg_edges = get_n_test_edges(train_graph, test_graph, train_ex, num_neg, board_ids)
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

