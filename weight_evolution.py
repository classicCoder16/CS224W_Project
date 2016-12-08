import random
import numpy


class EvolModel:

	def __init__(self):
		self.NUM_ITERS = 300
		self.NUM_CANDIDATES = 100
		self.w = None

	def fit(self, train_features, train_labels):
		num_features = train_features.shape[1]

		w = numpy.random.uniform(-1,1,num_features)
		for i in range(self.NUM_ITERS):
			candidates = self.generate_candidates(w)
			accuracies = self.get_accuracies(candidates, train_features, train_labels)
			index, val = max(enumerate(accuracies), key=lambda x:x[1])
			w = candidates[index]
		self.w = w

	def generate_candidates(self, w):
		candidates = []
		for i in range(self.NUM_CANDIDATES):
			c = numpy.random.normal(size=w.size)
			candidates.append(c + w)
		return candidates

	def get_accuracies(self, candidates, train_features, labels):
		accuracies = []
		for c in candidates:
			preds = numpy.sign(numpy.dot(train_features, c))
			accuracy = sum(preds == labels)
			accuracies.append(accuracy)
		return accuracies


	def predict(self, test_features):
		return numpy.sign(numpy.dot(test_features, self.w))

