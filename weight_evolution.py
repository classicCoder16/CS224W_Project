import random
import numpy

NUM_ITERS = 10
NUM_CANDIDATES = 5

def get_w(train_features, train_labels):
	labels = []
	f = open(train_labels, "r")
	for line in f:
		labels.append(int(line))
	f.close()

	training_set = []
	f = open(train_features, "r")
	for line in f:
		split_line = line.split(",")
		training_set.append(numpy.array([float(n.strip()) for n in split_line]))
	f.close()

	num_features = training_set[0].size

	w = numpy.random.uniform(-1,1,num_features)
	for i in range(NUM_ITERS):
		print w
		candidates = generate_candidates(w)
		accuracies = get_accuracies(candidates, training_set, labels)
		accuracy = -numpy.inf
		for i in range(len(accuracies)):
			if accuracies[i] > accuracy: w = candidates[i]
	return w

def generate_candidates(w):
	candidates = []
	for i in range(NUM_CANDIDATES):
		c = numpy.zeros(w.size)
		for i in range(w.size):
			c[i] = w[i] + numpy.random.normal()
		candidates.append(c)
	return candidates

def get_accuracies(candidates, training_set, labels):
	accuracies = []
	for c in candidates:
		accuracy = 0
		for i in range(len(labels)):
			if numpy.dot(c, training_set[i]) > 0:
				if labels[i] == 1: accuracy += 1
			else:
				if labels[i] == 0: accuracy += 1
		accuracies.append(accuracy / float(len(labels)))
	return accuracies


def test_w(test_features, test_labels, w):
	labels = []
	f = open(test_labels, "r")
	for line in f:
		labels.append(int(line))
	f.close()

	testing_set = []
	f = open(test_features, "r")
	for line in f:
		split_line = line.split(",")
		testing_set.append(numpy.array([float(n.strip()) for n in split_line]))
	f.close()

	accuracy = 0
	for i in range(len(labels)):
		if numpy.dot(w, testing_set[i]) > 0:
			if labels[i] == 1: accuracy += 1
		else:
			if labels[i] == 0: accuracy += 1

	return accuracy / float(len(labels))

def evolve_weights():
	w = get_w("train_features.txt", "train_labels.txt")
	return test_w("test_features.txt", "test_labels.txt", w)

evolve_weights()
