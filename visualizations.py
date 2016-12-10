import numpy
from sklearn.ensemble import RandomForestClassifier
#from weight_evolution import EvolModel
import pylab
import collections
import math
import datetime

def shortest_paths_misclassified(train_features_file, train_example_file, test_features_file, test_example_file, etype):
	rf = RandomForestClassifier(n_estimators=100)

	train_examples = numpy.load(train_features_file)
	train_zip = numpy.load(train_example_file)
	train_labels = [tup[1] for tup in train_zip]
	#train_pairs = [tup[0] for tup in train_zip]

	test_examples = numpy.load(test_features_file)
	test_zip = numpy.load(test_example_file)
	test_labels = [tup[1] for tup in test_zip]
	#test_pairs = [tup[0] for tup in test_zip]

	print 'Training model', rf
	rf.fit(train_examples, train_labels)
	preds = rf.predict(test_examples)

	print "Getting Shortest Paths"
	short_paths = []
	for i in range(preds.size):
		if preds[i] != test_labels[i]:
			short_paths.append(abs(train_examples[i][0]))

	pylab.figure()
	pylab.hist(short_paths, bins=numpy.arange(11) - 0.5)
	pylab.yscale('log')
	pylab.xlabel('Shortest Path')
	pylab.ylabel('Number of Edges')
	pylab.title('Shortest Paths Between Misclassified ' +etype+ ' Edges')
	pylab.show()

	pylab.figure()
	counts = collections.Counter(short_paths)
	print counts
	x = []
	y = []
	area = []
	for c in counts:
		x.append(c)
		y.append(counts[c]) 
		area.append(100*math.sqrt(counts[c]))
	pylab.scatter(x, y, s=area, c=numpy.random.rand(len(x)), alpha=.5)
	pylab.yscale('log')
	pylab.xlabel('Shortest Path')
	pylab.ylabel('Number of Edges')
	pylab.title('Shortest Paths Between Misclassified ' +etype+ ' Edges')
	pylab.show()

shortest_paths_misclassified("train_four_pin_features.npy", "train_four_pin_examples.npy", "test_four_pin_features.npy", "test_four_pin_examples.npy", "Pinned")
shortest_paths_misclassified("train_four_fol_features.npy", "train_four_fol_examples.npy", "test_four_fol_features.npy", "test_four_fol_examples.npy", "Follows")

def false_negative_timestamps(train_features_file, train_example_file, test_features_file, test_example_file, attribute_file, etype):
	rf = RandomForestClassifier(n_estimators=100)

	attr = numpy.load(attribute_file).item()



	train_examples = numpy.load(train_features_file)
	train_zip = numpy.load(train_example_file)
	train_labels = [tup[1] for tup in train_zip]
	#train_pairs = [tup[0] for tup in train_zip]

	test_examples = numpy.load(test_features_file)
	test_zip = numpy.load(test_example_file)
	test_labels = [tup[1] for tup in test_zip]
	test_pairs = [tup[0] for tup in test_zip]

	print 'Training model', rf
	rf.fit(train_examples, train_labels)
	preds = rf.predict(test_examples)

	print "Getting Timestamps"
	timestamps = []
	for i in range(preds.size):
		if preds[i] < 0 and test_labels[i] > 0:
			time = attr[test_pairs[i]]['pin_time']
			timestamps.append(datetime.datetime.fromtimestamp(time).isocalendar()[1])

	counts = collections.Counter(timestamps)
	counts = sorted(counts.items())
	x = []
	y = []
	for c in counts:
		x.append(c[0])
		y.append(c[1]) 

	min_week = min(x)
	x = [w-min_week for w in x]

	pylab.figure()
	pylab.plot(x, y)
	pylab.xlabel('Weeks In the Future')
	pylab.ylabel('Number of Edges')
	pylab.title('Failure to Predict ' +etype+ ' Edges Over Time')
	pylab.show()


false_negative_timestamps("train_temp_pin_features.npy", "train_temp_pin_examples.npy", "test_temp_pin_features.npy", "test_temp_pin_examples.npy", "smallest_test_attr.npy", "Pinned")

def evolution_heat_map(train_features_file, train_example_file, test_features_file, test_example_file, etype):
	bliss_model = EvolModel()

	train_examples = numpy.load(train_features_file)
	train_zip = numpy.load(train_example_file)
	train_labels = [tup[1] for tup in train_zip]
	#train_pairs = [tup[0] for tup in train_zip]

	test_examples = numpy.load(test_features_file)
	test_zip = numpy.load(test_example_file)
	test_labels = [tup[1] for tup in test_zip]

	print 'Training model', bliss
	bliss.fit(train_examples, train_labels)
	w = bliss.w





