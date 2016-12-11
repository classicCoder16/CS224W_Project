import numpy
from sklearn.ensemble import RandomForestClassifier
from weight_evolution import EvolModel
import pylab
import collections
import math
import datetime

def shortest_paths_misclassified(train_features_file, train_example_file, test_features_file, test_example_file, etype):
	rf = RandomForestClassifier(n_estimators=100)

	train_examples = numpy.load(train_features_file)

	train_zip = numpy.load(train_example_file)
	train_labels = [tup[1] for tup in train_zip]

	test_examples = numpy.load(test_features_file)
	test_zip = numpy.load(test_example_file)
	test_labels = [tup[1] for tup in test_zip]

	print 'Training model', rf
	rf.fit(train_examples, train_labels)
	preds = rf.predict(test_examples)

	print "Getting Shortest Paths"
	short_paths = []
	short_paths_false_pos = []
	short_paths_false_neg = []
	for i in range(preds.size):			
		if preds[i] == -1 and test_labels[i] == 1:
			dist = abs(train_examples[i][0])
			if dist == 1: dist = 11.0
			short_paths.append(dist)
			short_paths_false_neg.append(dist)
		elif preds[i] == 1 and test_labels[i] == -1:
			dist = abs(train_examples[i][0])
			if dist == 1: dist = 11.0
			short_paths.append(dist)
			short_paths_false_pos.append(dist)

	counts = collections.Counter(short_paths)
	counts_neg = collections.Counter(short_paths_false_neg)
	counts_pos = collections.Counter(short_paths_false_pos)
	print counts, counts_neg, counts_pos
	x = []
	y = []
	y_neg = []
	y_pos = []
	for i in range(1, 12):
		x.append(i)
		if i in counts: 
			y.append(counts[float(i)])
		else:
			y.append(0) 
		if i in counts_neg:
			y_neg.append(counts_neg[float(i)]) 
		else:
			y_neg.append(0)
		if i in counts_pos:
			y_pos.append(counts_pos[float(i)])
		else:
			y_pos.append(0)

	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'No Path']

	f = pylab.figure()
	ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
	ax.bar(x, y, align='center')
	ax.set_yscale('log')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.set_xlabel('Shortest Path')
	ax.set_ylabel('Number of Edges')
	ax.set_title('Shortest Paths Between Misclassified ' +etype+ ' Edges')
	f.show()
	
	
	f = pylab.figure()
	ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
	ax.bar(x, y_neg, align='center')
	ax.set_yscale('log')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.set_xlabel('Shortest Path')
	ax.set_ylabel('Number of Edges')
	ax.set_title('Shortest Paths Between False Negative ' +etype+ ' Edges')
	f.show()

	f = pylab.figure()
	ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
	ax.bar(x, y_pos, align='center')
	ax.set_yscale('log')
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.set_xlabel('Shortest Path')
	ax.set_ylabel('Number of Edges')
	ax.set_title('Shortest Paths Between False Positive ' +etype+ ' Edges')
	f.show()
	
shortest_paths_misclassified("train_four_pin_features.npy", "train_four_pin_examples.npy", "test_four_pin_features.npy", "test_four_pin_examples.npy", "Pinned")
shortest_paths_misclassified("train_four_fol_features.npy", "train_four_fol_examples.npy", "test_four_fol_features.npy", "test_four_fol_examples.npy", "Follows")

def false_negative_timestamps(train_features_file, train_example_file, test_features_file, test_example_file, attribute_file, etype):
	rf = RandomForestClassifier(n_estimators=100)

	attr = numpy.load(attribute_file).item()

	train_examples = numpy.load(train_features_file)
	train_zip = numpy.load(train_example_file)
	train_labels = [tup[1] for tup in train_zip]

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
false_negative_timestamps("train_temp_fol_features.npy", "train_temp_fol_examples.npy", "test_temp_fol_features.npy", "test_temp_fol_examples.npy", "smallest_test_attr.npy", "Follows")

def evolution_heat_map(train_features_file, train_example_file, test_features_file, test_example_file, etype):
	bliss = EvolModel()

	train_examples = numpy.load(train_features_file)
	train_zip = numpy.load(train_example_file)
	train_labels = [tup[1] for tup in train_zip]

	test_examples = numpy.load(test_features_file)
	test_zip = numpy.load(test_example_file)
	test_labels = [tup[1] for tup in test_zip]

	print 'Training model', bliss
	bliss.fit(train_examples, train_labels)
	w = bliss.w

	mat_len = w.size
	w = numpy.append(w, w).reshape(2, mat_len)

	print w

	pylab.imshow(w, cmap=pylab.get_cmap("Blues"))
	pylab.title('Bliss\'s Evolutionary Algorithm' +etype+ ' Feature Weights')
	pylab.xlabel('Features (increasing with time)')
	pylab.show()

evolution_heat_map("train_temp_pin_features.npy", "train_temp_pin_examples.npy", "test_temp_pin_features.npy", "test_temp_pin_examples.npy", "Pinned")
evolution_heat_map("train_temp_fol_features.npy", "train_temp_fol_examples.npy", "test_temp_fol_features.npy", "test_temp_fol_examples.npy", "Follows")







