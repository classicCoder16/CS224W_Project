import sklearn.preprocessing
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
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from weight_evolution import EvolModel


def test_classifiers(train_examples, train_labels, test_examples, test_labels):
	knn = KNeighborsClassifier()
	logistic = LogisticRegression()
	rf = RandomForestClassifier(n_estimators=100)
	my_nn = MLPClassifier(hidden_layer_sizes = (100, 50, 50))
	bliss_model = EvolModel()
	models = [bliss_model, knn, logistic, rf, my_nn]
	for model in models:
		print ''
		print 'Training model', model
		model.fit(train_examples, train_labels)
		preds = model.predict(test_examples)
		gt = np.array(test_labels)
		print 'Testing Set Results:'
		print 'Accuracy:', sklearn.metrics.accuracy_score(gt, preds)
		print 'Precision:', sklearn.metrics.precision_score(gt, preds)
		print 'Recall:', sklearn.metrics.recall_score(gt, preds)
		print 'F1 Score:', sklearn.metrics.f1_score(gt, preds)
		print ''

		print 'Training Set Results:'
		preds = model.predict(train_examples)
		gt_train = np.array(train_labels)
		print 'Accuracy:', sklearn.metrics.accuracy_score(gt_train, preds)
		print 'Precision:', sklearn.metrics.precision_score(gt_train, preds)
		print 'Recall:', sklearn.metrics.recall_score(gt_train, preds)
		print 'F1 Score:', sklearn.metrics.f1_score(gt_train, preds)
		print ''


def get_features(max_scc, examples, func):
	all_features = []
	for ex in examples:
		result = func(max_scc, ex[0], ex[1])
		all_features.append(result)
	return all_features

def get_all_features(feature_funcs, train_graph, train_examples, test_examples):
	all_train_features = []
	all_test_features = []
	for func in feature_funcs:
		print 'Extracting features with', func
		all_train_features.append(get_features(train_graph, train_examples, func))
		all_test_features.append(get_features(train_graph, test_examples, func))
	all_train_features = sklearn.preprocessing.scale(np.array(all_train_features).T)
	all_test_features = sklearn.preprocessing.scale(np.array(all_test_features).T)
	return all_train_features, all_test_features


def test_func(test_examples, test_labels, train_graph, func, num_pos):
	original_preds = []
	for i, cand in enumerate(test_examples):
		if (i%500 == 0): print i
		score = func(train_graph, cand[0], cand[1])
		original_preds.append(score)
	preds = sorted(zip(original_preds, test_labels), reverse=True)
	final_preds = []
	for i, elem in enumerate(preds):
		if i < num_pos: final_preds.append(1)
		else: final_preds.append(-1)
	preds, gt = map(list, zip(*preds))
	print ''
	print 'Accuracy:', sklearn.metrics.accuracy_score(gt, final_preds)
	print 'Precision:', sklearn.metrics.precision_score(gt, final_preds)
	print 'Recall:', sklearn.metrics.recall_score(gt, final_preds)
	print 'F1 Score:', sklearn.metrics.f1_score(gt, final_preds)


def validate_train(train_examples, train_labels, graph):
	print 'Validating Training Examples'
	i = 0
	for src_id, dst_id in train_examples:
		if graph.IsEdge(src_id, dst_id) and train_labels[i] == -1:
			print 'Conflict!'
		if not graph.IsEdge(src_id, dst_id) and train_labels[i] == 1:
			print 'Conflict!'
		i += 1

def validate_test(test_examples, test_labels, train_examples, test_graph, train_graph):
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
		i += 1


def main(train_file, test_file, output_root):

	# Load edges from files
	print 'Loading from files...'
	train_graph = snap.LoadEdgeList(snap.PUNGraph, train_file, 0, 1)
	test_graph = snap.LoadEdgeList(snap.PUNGraph, test_file, 0, 1)

	# Sample training examples
	print 'Extracting training examples'
	train_examples, train_labels = extract_examples(train_graph, 5000, 5000)
	validate_train(train_examples, train_labels, train_graph)

	# Get test examples
	print 'extracting testing esxamples'
	test_examples, test_labels = extract_test_examples(train_graph, test_graph, \
														train_examples, 2500, 2500)
	validate_test(test_examples, test_labels, train_examples, test_graph, train_graph)

	feature_funcs = [get_graph_distance, get_common_neighbors, jaccard_coefficient, adamic_adar,\
						preferential_attachment, get_degree_sum, get_coeff_sum, get_2_hops, \
						get_page_rank_sum, mean_nbr_deg_sum]

	for func in feature_funcs:
		print 'Testing', func
		test_func(test_examples, test_labels, train_graph, func, 2500)
	
	all_train_features, all_test_features = get_all_features(feature_funcs, train_graph, train_examples, test_examples)
	try:
		np.save('train_' + output_root + '_features', all_train_features)
		np.save('test_' + output_root + '_features', all_test_features)
		np.save('train_' + output_root + '_examples', zip(train_examples, train_labels))
		np.save('test_' + output_root + '_examples', zip(test_examples, test_labels))
	except Exception as e:
		print str(e)
	test_classifiers(all_train_features, train_labels, all_test_features, test_labels)


if __name__=='__main__':
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	output_root = sys.argv[3]
	main(train_file, test_file, output_root)
