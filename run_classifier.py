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

def test_classifiers(train_examples, train_labels, test_examples, test_labels):
	knn = KNeighborsClassifier()
	logistic = LogisticRegression()
	rf = RandomForestClassifier(n_estimators=100)
	my_svm = LinearSVC()
	models = [knn, logistic, rf, my_svm]
	for model in models:
		print 'Training model', model
		model.fit(train_examples, train_labels)
		preds = model.predict(test_examples)
		print 'Accuracy:', sklearn.metrics.accuracy_score(test_labels, preds)
		print 'Precision:', sklearn.metrics.precision_score(test_labels, preds)
		print 'Recall:', sklearn.metrics.recall_score(test_labels, preds)
		print 'F1 Score:', sklearn.metrics.f1_score(test_labels, preds)



def get_features(max_scc, examples, func):
	all_features = []
	for ex in examples:
		result = func(max_scc, ex[0], ex[1])
		all_features.append(result)
	return all_features

def test_func(test_examples, test_labels, max_scc, func, num_pos):
	original_preds = []
	for i, cand in enumerate(test_examples):
		if (i%20 == 0): print i
		score = func(max_scc, cand[0], cand[1])
		original_preds.append(score)
	preds = sorted(zip(original_preds, test_labels), reverse=True)
	final_preds = []
	for i, elem in enumerate(preds):
		if i < num_pos: final_preds.append(1)
		else: final_preds.append(-1)
	preds, gt = map(list, zip(*preds))
	print 'Accuracy:', sklearn.metrics.accuracy_score(gt, final_preds)
	print 'Precision:', sklearn.metrics.precision_score(gt, final_preds)
	print 'Recall:', sklearn.metrics.recall_score(gt, final_preds)
	print 'F1 Score:', sklearn.metrics.f1_score(gt, final_preds)
	return original_preds


def main():
	print 'Extracting training examples.'
	train_graph_obj = Train_Graph(graph_file_root='smallest_train')
	train_pgraph = train_graph_obj.pgraph
	max_scc = train_pgraph
	train_examples, train_labels = extract_examples(max_scc, 10000, 10000)

	'''
	We need to make sure that every pair of nodes actually appears in the
	original training component, but every pair itself is not in training
	examples.
	'''
	print 'Extracting testing examples'
	test_graph_obj = Test_Graph(graph_file_root='smallest_test')
	test_pgraph = test_graph_obj.pgraph
	test_examples, test_labels = extract_test_examples(max_scc, test_pgraph, \
														train_examples, 5000, 5000)

	feature_funcs = [preferential_attachment, get_2_hops, \
					get_degree_sum, get_coeff_sum, get_graph_distance]
	# for func in feature_funcs:
	# 	print 'Testing', func
	# 	test_func(test_examples, test_labels, max_scc, func, 5000)

	all_train_features = []
	all_test_features = []
	for func in feature_funcs:
		print 'Extracting features with', func
		all_train_features.append(get_features(max_scc, train_examples, func))
		all_test_features.append(get_features(max_scc, test_examples, func))
	all_train_features = np.array(all_train_features).T
	all_test_features = np.array(all_test_features).T
	test_classifiers(all_train_features, train_labels, all_test_features, test_labels)












if __name__=='__main__':
	main()