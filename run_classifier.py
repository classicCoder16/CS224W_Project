import sklearn.metrics
import numpy as np
import snap
from train_graph import Train_Graph
from test_graph import Test_Graph
from get_examples import *
from feature_extractors import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def print_metrics(gt, pred):
	print 'Accuracy:', sklearn.metrics.accuracy_score(gt, pred)
	print 'Precision:', sklearn.metrics.precision_score(gt, pred)
	print 'Recall:', sklearn.metrics.recall_score(gt, pred)
	print 'F1 Score:', sklearn.metrics.f1_score(gt, pred)

def test_classifiers(train_examples, train_labels, test_examples, test_labels, edge_types):
	knn = KNeighborsClassifier()
	logistic = LogisticRegression()
	rf = RandomForestClassifier(n_estimators=100)
	my_nn = MLPClassifier(hidden_layer_sizes = (100, 50))
	models = [knn, logistic, rf, my_nn]
	for model in models:
		print 'Training model', model
		model.fit(train_examples, train_labels)
		preds = model.predict(test_examples)
		preds1 = [elem for i, elem in enumerate(preds) if edge_types[i] == 'follows']
		gt1 = [elem for i, elem in enumerate(test_labels) if edge_types[i] == 'follows']
		preds2 = [elem for i, elem in enumerate(preds) if edge_types[i] == 'pinned']
		gt2 = [elem for i, elem in enumerate(test_labels) if edge_types[i] == 'pinned']
		print ''
		print 'Evaluating "follows":'
		print_metrics(gt1, preds1)
		print ''
		print 'Evaluating "pinned":'
		print_metrics(gt2, preds2)
		# print 'Training Set Results:'
		# preds = model.predict(train_examples)
		# gt_train = np.array(train_labels)
		# print 'Accuracy:', sklearn.metrics.accuracy_score(gt_train, preds)
		# print 'Precision:', sklearn.metrics.precision_score(gt_train, preds)
		# print 'Recall:', sklearn.metrics.recall_score(gt_train, preds)
		# print 'F1 Score:', sklearn.metrics.f1_score(gt_train, preds)
		# print ''

def test_proximity(feature_funcs, test_examples, test_labels, max_scc, num_pos, test_edge_types):
	for func in feature_funcs:
		print 'Testing', func
		test_func(test_examples, test_labels, max_scc, func, num_pos, test_edge_types)

def get_all_features(feature_funcs, max_scc, train_examples, test_examples):
	all_train_features = []
	all_test_features = []
	for func in feature_funcs:
		print 'Extracting features with', func
		all_train_features.append(get_features(max_scc, train_examples, func))
		all_test_features.append(get_features(max_scc, test_examples, func))
	all_train_features = np.array(all_train_features).T
	all_test_features = np.array(all_test_features).T
	return all_train_features, all_test_features

def get_features(max_scc, examples, func):
	all_features = []
	for ex in examples:
		result = func(max_scc, ex[0], ex[1])
		all_features.append(result)
	return all_features

def test_func(test_examples, test_labels, max_scc, func, num_pos, test_edge_types):
	original_preds = []
	# For every input
	for i, cand in enumerate(test_examples):
		if (i%500 == 0): print i
		score = func(max_scc, cand[0], cand[1])
		# Append the score to score list
		original_preds.append(score)
	# Define a list of (score, label, edge_type), sorted by score
	preds = sorted(zip(original_preds, test_labels, test_edge_types), reverse=True)
	final_preds_1 = []
	final_preds_2 = []
	final_preds = []
	gt_1 = []
	gt_2 = []
	gt = []
	for i, elem in enumerate(preds):
		score, label, edge_type = elem
		if edge_type == 'follows':
			if i < num_pos: final_preds_1.append(1)
			else: final_preds_1.append(-1)
			gt_1.append(label)
		else: 
			if i < num_pos: final_preds_2.append(1)
			else: final_preds_2.append(-1)
			gt_2.append(label)
		if i < num_pos:
			final_preds.append(1)
		else:
			final_preds.append(-1)
		gt.append(label)
	print ''
	print 'Evaluating "follows":'
	print_metrics(gt_1, final_preds_1)

	print ''
	print 'Evaluating "pinned":'
	print_metrics(gt_2, final_preds_2)

	print ''
	print 'Evaluating Both:'
	print_metrics(gt, final_preds)

def validate_train(train_examples, train_labels, graph):
	print 'Validating Training Examples'
	i = 0
	for src_id, dst_id in train_examples:
		if graph.IsEdge(src_id, dst_id) and train_labels[i] == -1:
			print 'Conflict!'
		if not graph.IsEdge(src_id, dst_id) and train_labels[i] == 1:
			print 'Conflict!'
		i += 1


def isBoard(n, board_node_ids):
	return n >= board_node_ids[0] and n <= board_node_ids[1]


def validate_test(test_examples, test_labels, train_examples, test_graph, train_graph, board_node_ids):
	print 'Validating Testing Examples'
	i = 0
	for src_id, dst_id in test_examples:
		if test_graph.IsEdge(src_id, dst_id) and test_labels[i] == -1:
			print 'Conflict!'
		if not test_graph.IsEdge(src_id, dst_id) and test_labels[i] == 1:
			print 'Conflict!'
		if test_examples[i] in train_examples:
			print 'Conflict!'
		if not train_graph.IsNode(src_id) or not train_graph.IsNode(dst_id):
			print 'Conflict!'
		if isBoard(src_id, board_node_ids) and isBoard(dst_id, board_node_ids):
			print 'Conflict!'
		if not isBoard(src_id, board_node_ids) and not isBoard(dst_id, board_node_ids):
			print 'Conflict!'
		i += 1

def get_edge_types(examples, board_ids):
	edge_types = []
	for src_id, dst_id in examples:
		if src_id < board_ids[0] or dst_id < board_ids[0]:
			edge_types.append('follows')
		else:
			edge_types.append('pinned')
	print edge_types[:50]
	return edge_types

def main():
	print 'Extracting training examples.'
	train_graph_obj = Train_Graph(graph_file_root='smallest_train')
	train_pgraph = train_graph_obj.pgraph
	max_scc = train_pgraph
	train_examples, train_labels = extract_examples(max_scc, 10000, 10000)
	validate_train(train_examples, train_labels, max_scc)


	'''
	We need to make sure that every pair of nodes actually appears in the
	original training component, but every pair itself is not in training
	examples.
	'''
	print 'Extracting testing examples'
	test_graph_obj = Test_Graph(graph_file_root='smallest_test')
	test_pgraph = test_graph_obj.pgraph
	test_examples, test_labels = extract_test_examples(max_scc, test_pgraph, \
								train_examples, 5000, 5000, test_graph_obj.board_node_ids)
	print 'Getting Edge types'
	test_edge_types = get_edge_types(test_examples, test_graph_obj.board_node_ids)

	# Make sure test set satisfies criteria
	validate_test(test_examples, test_labels, train_examples, test_pgraph, max_scc, test_graph_obj.board_node_ids)

	# Define all feature functions we will be using
	feature_funcs = [get_ev_centr_sum, get_page_rank_sum, preferential_attachment, \
					get_2_hops, get_degree_sum, std_nbr_degree_sum, \
					mean_nbr_deg_sum, adamic_adar_2, common_neighbors_2, \
					jaccard_2]
	# feature_funcs = [preferential_attachment]

	# Test each feature function on its own
	test_proximity(feature_funcs, test_examples, test_labels, max_scc, 5000, test_edge_types)

	# Convert our training examples and testing examples to feature
	# vectors
	all_train_features, all_test_features = get_all_features(feature_funcs, max_scc, train_examples, test_examples)
	
	# Test our classifiers over these features
	test_classifiers(all_train_features, train_labels, all_test_features, test_labels, test_edge_types)












if __name__=='__main__':
	main()