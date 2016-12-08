import random
import numpy


class EvolModel:

	def __init__(self):
		self.NUM_ITERS = 300
		self.NUM_CANDIDATES = 100
		self.w = None

	def fit(self, train_features, train_labels):
		# labels = []
		# f = open(train_labels, "r")
		# for line in f:
		# 	labels.append(int(line))
		# f.close()

		# training_set = []
		# f = open(train_features, "r")
		# for line in f:
		# 	training_set.append(numpy.array([float(n) for n in line.split(",")]))
		# f.close()

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
			# for i in range(w.size):
			# 	c[i] = w[i] + numpy.random.normal()
			candidates.append(c + w)
		return candidates

	def get_accuracies(self, candidates, train_features, labels):
		accuracies = []
		for c in candidates:
			preds = numpy.sign(numpy.dot(train_features, c))
			accuracy = sum(preds == labels)
			accuracies.append(accuracy)
			# accuracy = 0
			# for i in range(len(labels)):
			# 	if numpy.dot(c, training_set[i]) > 0:
			# 		if labels[i] == 1: accuracy += 1
			# 	else:
			# 		if labels[i] == 0: accuracy += 1
			# accuracies.append(accuracy / float(len(labels)))
		return accuracies


	def predict(self, test_features, test_labels):
		# labels = []
		# f = open(test_labels, "r")
		# for line in f:
		# 	labels.append(int(line))
		# f.close()

		# testing_set = []
		# f = open(test_features, "r")
		# for line in f:
		# 	split_line = line.split(",")
		# 	testing_set.append(numpy.array([float(n.strip()) for n in split_line]))
		# f.close()
		return numpy.sign(numpy.dot(test_features, self.w))

		# accuracy = 0
		# for i in range(len(labels)):
		# 	if numpy.dot(w, testing_set[i]) > 0:
		# 		if labels[i] == 1: accuracy += 1
		# 	else:
		# 		if labels[i] == 0: accuracy += 1

		# return accuracy / float(len(labels))

	# def evolve_weights():
	# 	w = get_w("train_features.txt", "train_labels.txt")
	# 	return test_w("test_features.txt", "test_labels.txt", w)

# evolve_weights()
