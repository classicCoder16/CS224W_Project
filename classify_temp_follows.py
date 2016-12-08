import sklearn.preprocessing
import sklearn.metrics
import sys
import datetime
from train_graph import Train_Graph
from test_graph import Test_Graph
from feature_extractors import *
import snap
import math
from get_examples import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from weight_evolution import EvolModel

def print_metrics(gt, pred):
	print 'Accuracy:', sklearn.metrics.accuracy_score(gt, pred)
	print 'Precision:', sklearn.metrics.precision_score(gt, pred)
	print 'Recall:', sklearn.metrics.recall_score(gt, pred)
	print 'F1 Score:', sklearn.metrics.f1_score(gt, pred)

def is_board(n, board_ids):
	return n >= board_ids[0] and n <= board_ids[1]

def is_pin(n, board_ids):
	return n > board_ids[1]

def is_user(n, board_ids):
	return n < board_ids[0]

def add_edges_from_int(graph, interval_edges, intervals):
	for i in intervals:
		for src_id, dst_id in interval_edges[i]:
			graph.AddEdge(src_id, dst_id)

def get_feat_vals(graph, examples, feature_funcs):
	result = np.zeros([len(examples), len(feature_funcs)])
	for i, elem in enumerate(examples):
		src_id, dst_id = elem
		for j, func in enumerate(feature_funcs):
			score = func(graph, src_id, dst_id)
			result[i][j] = score
	return result

def test_classifiers(train_examples, train_labels, test_examples, test_labels):
	knn = KNeighborsClassifier()
	logistic = LogisticRegression()
	rf = RandomForestClassifier(n_estimators=100)
	my_nn = MLPClassifier(hidden_layer_sizes = (100, 50, 50))
	bliss_model = EvolModel()
	models = [knn, logistic, rf, my_nn, bliss_model]
	for model in models:
		print 'Training model', model
		model.fit(train_examples, train_labels)
		preds = model.predict(test_examples)
		gt = [elem for elem in test_labels]
		print ''
		print 'Evaluating Testing Set:'
		print_metrics(gt, preds)

		print ''
		print 'Evaluating Training Set:'
		preds_train = model.predict(train_examples)
		gt_train = [elem for elem in train_labels]
		print_metrics(gt_train, preds_train)


def get_train_features(train_examples, graph, interval_edges, feature_funcs):
	print 'Nodes, Edges before:', graph.GetNodes(), graph.GetEdges()
	all_edges = [(edge.GetSrcNId(), edge.GetDstNId()) for edge in graph.Edges()]
	for src_id, dst_id in all_edges:
		graph.DelEdge(src_id, dst_id)
	add_edges_from_int(graph, interval_edges, [0])
	num_intervals = len(interval_edges)
	num_features = len(feature_funcs)
	num_examples = len(train_examples)

	# Old vals is a num_examples x num_features np.array
	print 'Getting init values'
	old_vals = get_feat_vals(graph, train_examples, feature_funcs)

	# Final feats will eventually be a num_examples x (num_intervals - 2)*num_features
	final_feats = np.zeros([num_examples, (num_intervals - 2)*num_features])
	for int_num in range(1, num_intervals - 1):
		print 'Evaluating Interval', int_num
		add_edges_from_int(graph, interval_edges, [int_num])
		new_vals = get_feat_vals(graph, train_examples, feature_funcs)
		deltas = new_vals - old_vals
		final_feats[:, (int_num - 1)*num_features:int_num*num_features] = deltas
		old_vals = new_vals
	add_edges_from_int(graph, interval_edges, [num_intervals - 1])
	print 'Nodes, edges afterwards:', graph.GetNodes(), graph.GetEdges()
	print final_feats.shape
	return final_feats

def get_test_features(test_examples, graph, interval_edges, feature_funcs):
	print 'Nodes, Edges before:', graph.GetNodes(), graph.GetEdges()
	all_edges = [(edge.GetSrcNId(), edge.GetDstNId()) for edge in graph.Edges()]
	print 'Deleting all edges...'
	for src_id, dst_id in all_edges:
		graph.DelEdge(src_id, dst_id)
	add_edges_from_int(graph, interval_edges, [0, 1])
	num_intervals = len(interval_edges)
	num_features = len(feature_funcs)
	num_examples = len(test_examples)

	# Old vals is a num_examples x num_features np.array
	old_vals = get_feat_vals(graph, test_examples, feature_funcs)

	# Final feats will eventually be a num_examples x (num_intervals - 2)*num_features
	final_feats = np.zeros([num_examples, (num_intervals - 2)*num_features])
	for int_num in range(2, num_intervals):
		print 'Evaluating Interval', int_num
		add_edges_from_int(graph, interval_edges, [int_num])
		new_vals = get_feat_vals(graph, test_examples, feature_funcs)
		deltas = new_vals - old_vals
		final_feats[:, (int_num - 2)*num_features:(int_num - 1)*num_features] = deltas
		old_vals = new_vals
	print final_feats.shape
	print 'Nodes, edges afterwards:', graph.GetNodes(), graph.GetEdges()
	return final_feats


def get_train_set(train_pgraph, interval_edges, board_ids, attributes, num_pos=10000, num_neg=10000):
	last_interval = interval_edges[-1]
	all_pinned_edges = []
	for src_id, dst_id in last_interval:
		if not is_user(src_id, board_ids) and not is_user(dst_id, board_ids): continue
		if 'create_time' in attributes[tuple(sorted([src_id, dst_id], reverse=True))]: continue
		all_pinned_edges.append(tuple(sorted([src_id, dst_id])))
	pos_edges = random.sample(all_pinned_edges, num_pos)
	pos_labels = [1]*len(pos_edges)

	neg_edges = get_neg_pin_edges(train_pgraph, num_neg, board_ids)
	neg_labels = [-1]*len(neg_edges)
	all_pairs = pos_edges + neg_edges 
	all_labels = pos_labels + neg_labels

	return all_pairs, all_labels



def get_time_limits(graph_obj):
	attributes = graph_obj.attributes
	max_time = datetime.datetime(1, 1, 1)
	min_time = datetime.datetime(3000, 12, 31)
	for edge in attributes:
		if not isinstance(edge, tuple): continue
		if 'pin_time' in attributes[edge]: key_val = 'pin_time'
		if 'follow_time' in attributes[edge]: key_val = 'follow_time'
		if 'create_time' in attributes[edge]: key_val = 'create_time'
		time_val = datetime.datetime.fromtimestamp(attributes[edge][key_val])
		if time_val < min_time: min_time = time_val
		if time_val > max_time: max_time = time_val
	print min_time, max_time
	return min_time, max_time
'''
Return a list of num_intervals lists, where each sub-list holds
the edges formed during that interval.
'''
def get_intervals(min_time, max_time, graph, attributes, num_intervals, board_ids):
	int_edges = [[] for i in range(num_intervals)]
	time_delta = (max_time - min_time)/num_intervals
	all_edges = [(edge.GetSrcNId(), edge.GetDstNId()) for edge in graph.Edges()]
	for src_id, dst_id in all_edges:
		if src_id < board_ids[0] or src_id > board_ids[1]:
			key_val = (dst_id, src_id)
		else:
			key_val = (src_id, dst_id)
		if 'pin_time' in attributes[key_val]: 
			time_val = attributes[key_val]['pin_time']
		elif 'follow_time' in attributes[key_val]: 
			time_val = attributes[key_val]['follow_time']
		elif 'create_time' in attributes[key_val]: 
			time_val = attributes[key_val]['create_time']
		else:
			print 'Here!'
			continue
		time_val = datetime.datetime.fromtimestamp(time_val)
		index = int(math.ceil((time_val - min_time).total_seconds()/time_delta.total_seconds()) - 1)
		if index == (num_intervals - 1):
			print 'Why are we here?'
		index = min(index, num_intervals - 1)
		int_edges[index].append((src_id, dst_id))
	for interval in int_edges:
		print len(interval)
	return int_edges

def add_init_edges(train_pgraph, int_edges):
	for src_id, dst_id in int_edges[0]:
		train_pgraph.AddEdge(src_id, dst_id)

def main(input_train, input_test, num_intervals):
	# Read in the graph
	train_graph_obj = Train_Graph(graph_file_root=input_train)
	train_pgraph = train_graph_obj.pgraph
	# (Get max SCC?)

	# Get limits on the time range
	print 'Getting time limits'
	min_time, max_time = get_time_limits(train_graph_obj)

	# Divide into intervals based on time range
	print 'Dividing into intervals'
	interval_edges = get_intervals(min_time, max_time, train_pgraph, \
			train_graph_obj.attributes, num_intervals, train_graph_obj.board_node_ids)
	assert sum([len(interval) for interval in interval_edges]) == train_pgraph.GetEdges()
	
	# Extract positive and negative training examples in the last frame
	print 'Getting training examples/labels'
	train_examples, train_labels = get_train_set(train_pgraph, interval_edges, \
								train_graph_obj.board_node_ids, train_graph_obj.attributes)

	# Contruct our testing set
	test_graph_obj = Test_Graph(graph_file_root=input_test)
	test_pgraph = test_graph_obj.pgraph
	print 'Getting testing examples/labels'
	test_examples, test_labels = get_pin_tst_ex(train_pgraph, test_pgraph, \
								train_examples, 5000, 5000, test_graph_obj.board_node_ids)

	feature_funcs = [get_graph_distance, get_ev_centr_sum, get_page_rank_sum, \
					preferential_attachment, get_2_hops, get_degree_sum, \
					std_nbr_degree_sum, mean_nbr_deg_sum, adamic_adar_2, \
					common_neighbors_2]
	print 'Extracting Training features...'
	train_features = get_train_features(train_examples, train_pgraph, interval_edges, feature_funcs)
	np.save('train_temp_fol_features', train_features)
	np.save('train_temp_fol_examples', zip(train_examples, train_labels))
	train_features = sklearn.preprocessing.scale(train_features)	
	print 'Extracting Testing features...'
	test_features = get_test_features(test_examples, train_pgraph, interval_edges, feature_funcs)
	np.save('test_temp_fol_features', test_features)
	np.save('test_temp_fol_examples', zip(test_examples, test_labels))
	test_features = sklearn.preprocessing.scale(test_features)	

	test_classifiers(train_features, train_labels, test_features, test_labels)


if __name__=='__main__':
	input_train = sys.argv[1]
	input_test = sys.argv[2]
	num_intervals = int(sys.argv[3])
	main(input_train, input_test, num_intervals)