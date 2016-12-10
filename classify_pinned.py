import sklearn.preprocessing
import sklearn.metrics
import numpy as np
import snap
import sys
from train_graph import Train_Graph
from test_graph import Test_Graph
from get_examples import *
from feature_extractors import *
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

def test_classifiers(train_examples, train_labels, test_examples, test_labels):
	knn = KNeighborsClassifier()
	logistic = LogisticRegression()
	rf = RandomForestClassifier(n_estimators=100)
	my_nn = MLPClassifier(hidden_layer_sizes = (100, 50, 50))
	bliss_model = EvolModel()
	models = [bliss_model, knn, logistic, rf, my_nn]
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


def test_proximity(feature_funcs, test_examples, test_labels, max_scc, num_pos):
	for func in feature_funcs:
		print 'Testing', func
		test_func(test_examples, test_labels, max_scc, func, num_pos)

def get_all_features(feature_funcs, max_scc, train_examples, test_examples):
	all_train_features = []
	all_test_features = []
	for func in feature_funcs:
		print 'Extracting features with', func
		all_train_features.append(get_features(max_scc, train_examples, func))
		all_test_features.append(get_features(max_scc, test_examples, func))
	# Transpose since sklearn takes nsamples, nfeatures shape
	all_train_features = np.array(all_train_features).T
	all_test_features = np.array(all_test_features).T
	return all_train_features, all_test_features

def get_features(max_scc, examples, func):
	all_features = []
	for ex in examples:
		result = func(max_scc, ex[0], ex[1])
		all_features.append(result)
	return all_features

def test_func(test_examples, test_labels, max_scc, func, num_pos):
	original_preds = []
	# For every input
	for i, cand in enumerate(test_examples):
		if (i%500 == 0): print i
		score = func(max_scc, cand[0], cand[1])
		# Append the score to score list
		original_preds.append(score)
	# Define a list of (score, label, edge_type), sorted by score
	preds = sorted(zip(original_preds, test_labels), reverse=True)
	final_preds = []
	gt = []
	for i, elem in enumerate(preds):
		score, label = elem
		if i < num_pos: final_preds.append(1)
		else: final_preds.append(-1)
		gt.append(label)

	print ''
	print 'Evaluating...:'
	print_metrics(gt, final_preds)

def validate_train(train_examples, train_labels, graph, board_ids):
	print 'Validating Training Examples'
	i = 0
	for src_id, dst_id in train_examples:
		if graph.IsEdge(src_id, dst_id) and train_labels[i] == -1:
			print 'Conflict!'
		if not graph.IsEdge(src_id, dst_id) and train_labels[i] == 1:
			print 'Conflict!'
		if not ((isBoard(src_id, board_ids) and isPin(dst_id, board_ids)) or \
			(isBoard(dst_id, board_ids) and isPin(src_id, board_ids))):
			print 'Conflict!'
		i += 1


def isBoard(n, board_node_ids):
	return n >= board_node_ids[0] and n <= board_node_ids[1]

def isPin(n, board_node_ids):
	return n > board_node_ids[1]


def validate_test(test_examples, test_labels, train_examples, test_graph, train_graph, board_ids):
	print 'Validating Testing Examples'
	i = 0
	for src_id, dst_id in test_examples:
		if test_graph.IsEdge(src_id, dst_id) and test_labels[i] == -1:
			print 'Conflict, edge in test graph has negative label!'
		if not test_graph.IsEdge(src_id, dst_id) and test_labels[i] == 1:
			print 'Conflict, positive edge not in test graph'
		if test_examples[i] in train_examples:
			print 'Conflict, edges in traning set!'
		if not train_graph.IsNode(src_id) or not train_graph.IsNode(dst_id):
			print 'Conflict, nodes not in train graph!'
		if not ((isBoard(src_id, board_ids) and isPin(dst_id, board_ids)) or \
			(isBoard(dst_id, board_ids) and isPin(src_id, board_ids))):
			print 'Conflict, edge type!'
		i += 1


def main(input_train, input_test, output_root):
	print 'Extracting training examples.'
	train_graph_obj = Train_Graph(graph_file_root=input_train)
	train_pgraph = train_graph_obj.pgraph
	max_scc = train_pgraph
	board_ids = train_graph_obj.board_node_ids
	train_examples, train_labels = get_pin_tr_ex(max_scc, 5000, 5000, board_ids)
	validate_train(train_examples, train_labels, max_scc, board_ids)

	'''
	We need to make sure that every pair of nodes actually appears in the
	original training component, but every pair itself is not in training
	examples.
	'''
	print 'Extracting testing examples'
	test_graph_obj = Test_Graph(graph_file_root=input_test)
	test_pgraph = test_graph_obj.pgraph
	test_examples, test_labels = get_pin_tst_ex(max_scc, test_pgraph, \
								train_examples, 2500, 2500, test_graph_obj.board_node_ids)

	# Make sure test set satisfies criteria
	validate_test(test_examples, test_labels, train_examples, test_pgraph, max_scc, test_graph_obj.board_node_ids)

	# Define all feature functions we will be using
	feature_funcs = [get_graph_distance, get_ev_centr_sum, get_page_rank_sum, \
					preferential_attachment, get_2_hops, get_degree_sum, \
					std_nbr_degree_sum, mean_nbr_deg_sum, adamic_adar_2, \
					common_neighbors_2]
	# feature_funcs = [jaccard_2, preferential_attachment, get_degree_sum]

	# # Test each feature function on its own
	# test_proximity(feature_funcs, test_examples, test_labels, max_scc, 5000)

	# Convert our training examples and testing examples to feature
	# vectors
	print 'Extracting features for classifier'
	all_train_features, all_test_features = get_all_features(feature_funcs, max_scc, train_examples, test_examples)
	print 'Saving features to file...'
	try:
		np.save('train_' + output_root + '_pin_features', all_train_features)
		np.save('test_' + output_root + '_pin_features', all_test_features)
		np.save('train_' + output_root + '_pin_examples', zip(train_examples, train_labels))
		np.save('test_' + output_root + '_pin_examples', zip(test_examples, test_labels))
	except Exception as e:
		print str(e)
	all_train_features = sklearn.preprocessing.scale(all_train_features)
	all_test_features = sklearn.preprocessing.scale(all_test_features)
	# # Test our classifiers over these features
	test_classifiers(all_train_features, train_labels, all_test_features, test_labels)




if __name__=='__main__':
	input_train = sys.argv[1]
	input_test = sys.argv[2]
	output_root = sys.argv[3]
	main(input_train, input_test, output_root)

