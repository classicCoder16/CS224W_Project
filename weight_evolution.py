import random
import numpy


class EvolModel:

	def __init__(self):
		self.NUM_ITERS = 500
		self.NUM_CANDIDATES = 500
		self.w = None

	def fit(self, train_features, train_labels):
		num_features = train_features.shape[1]
		train_labels = numpy.array(train_labels)
		w = numpy.random.uniform(-5,5,num_features)
		prev_acc = 0.0
		for i in range(self.NUM_ITERS):
			candidates = self.generate_candidates(w)
			accuracies = self.get_accuracies(candidates, train_features, train_labels)
			index, val = max(enumerate(accuracies), key=lambda x:x[1])
			if val >= prev_acc:
				w = candidates[index]
				prev_acc = val
		self.w = w

	def generate_candidates(self, w):
		candidates = []
		# for i in range(self.NUM_CANDIDATES):
		c = numpy.random.uniform(-2, 2, size=[self.NUM_CANDIDATES, w.size])
		candidates = c + w
		return candidates

	def get_accuracies(self, candidates, train_features, labels):
		# accuracies = []
		# for i, c in enumerate(candidates):
		# 	print i
		preds = numpy.sign(numpy.dot(train_features, candidates.T))
		matches = preds.T*labels
		accuracies = numpy.sum(matches > 0, axis = 1).flatten().tolist()
		return accuracies


	def predict(self, test_features):
		return numpy.sign(numpy.dot(test_features, self.w))

