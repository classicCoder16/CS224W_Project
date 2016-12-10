import snap
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from weight_evolution import EvolModel
from train_graph import Train_Graph
from test_graph import Test_Graph
from feature_extractors import *

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

def main(temp_train_feats, temp_train_ex, temp_test_feats, temp_test_ex, graph_file):
	train_graph_obj = Train_Graph(graph_file_root=graph_file)
	graph = train_graph_obj.pgraph
	train_examples = temp_train_ex[:, 0].tolist()
	train_labels = temp_train_ex[:, 1].tolist()
	test_examples = temp_test_ex[:, 0].tolist()
	test_labels = temp_test_ex[:, 1].tolist()
	feature_funcs = [preferential_attachment]
	# feature_funcs = [get_graph_distance, get_ev_centr_sum, get_page_rank_sum, \
	# 				preferential_attachment, get_2_hops, get_degree_sum, \
	# 				std_nbr_degree_sum, mean_nbr_deg_sum, adamic_adar_2, \
	# 				common_neighbors_2]
	print 'Extracting features'
	norm_train_features, norm_test_features = get_all_features(feature_funcs, graph, train_examples, test_examples)
	all_train_feats = np.hstack([norm_train_features, temp_train_feats])
	all_test_feats = np.hstack([norm_test_features, temp_test_feats])
	all_train_feats = sklearn.preprocessing.scale(all_train_feats)
	all_test_feats = sklearn.preprocessing.scale(all_test_feats)
	print 'Testing Classifiers'
	test_classifiers(all_train_features, train_labels, all_test_features, test_labels)

if __name__=='__main__':
	temp_train_feats = np.load(sys.argv[1])
	temp_train_ex = np.load(sys.argv[2])
	temp_test_feats = np.load(sys.argv[3])
	temp_test_ex = np.load(sys.argv[4])
	graph_file = np.load(sys.argv[5])
	main(temp_train_feats, temp_train_ex, temp_test_feats, temp_test_ex, graph_file)