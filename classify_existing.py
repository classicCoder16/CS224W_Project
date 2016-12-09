import sklearn.preprocessing
import sklearn.metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from weight_evolution import EvolModel
import sys

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

def main(train_ex_file, train_feats_file, test_ex_file, test_feats_files):
	train_features = sklearn.preprocessing.scale(np.load(train_feats_file))
	test_features = sklearn.preprocessing.scale(np.load(test_feats_files))
	train_examples, train_labels = zip(*np.load(train_ex_file))
	test_examples, test_labels = zip(*np.load(test_ex_file))
	test_classifiers(train_features, train_labels, test_features, test_labels)



if __name__=='__main__':
	train_ex_file = sys.argv[1]
	train_feats_file = sys.argv[2]
	test_ex_file = sys.argv[3]
	test_feats_files = sys.argv[4]
	main(train_ex_file, train_feats_file, test_ex_file, test_feats_files)
