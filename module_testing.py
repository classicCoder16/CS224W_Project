'''
This file will create small training and testing graphs,
extract examples from each of them, extract features from the
examples, and run a classifier on it. 
'''
import sys
import datetime
from train_graph import Train_Graph
from test_graph import Test_Graph
from feature_extractors import *
from get_examples import *

def test_graph_loading(arguments):
	print 'Testing graph loading...'
	train_graph_obj = Train_Graph(graph_file_root='mid_train')
	print 'Done!'
	print train_graph_obj.attributes.items()[:10]
	print train_graph_obj.pgraph.GetNodes()
	print train_graph_obj.pgraph.GetEdges()

def test_graph_creation(arguments):
	# Arguments are data_path, output_root
	print arguments
	print 'Testing train_graph.py...'
	time_lbound = datetime.datetime(2013, 1, 1)
	time_ubound = datetime.datetime(2013, 4, 30)
	train_graph_obj = Train_Graph(time_lbound=time_lbound, \
								  time_ubound=time_ubound, \
								  src_path=arguments[0])

	train_graph_obj.write_to_file(arguments[1])
	print train_graph_obj.attributes.items()[:10]
	print train_graph_obj.pgraph.GetNodes()
	print train_graph_obj.pgraph.GetEdges()
	print 'Done!'

def test_testing_graph_creation(arguments):
	# Arguments are data location, training graph file root, output file root
	time_lbound = datetime.datetime(2013, 5, 1)
	time_ubound = datetime.datetime(2013, 6, 30)
	print 'Testing test_graph.py...'
	test_graph_obj = Test_Graph(time_lbound=time_lbound, \
								  time_ubound=time_ubound, \
								  src_path=arguments[0], \
								  node_file_root=arguments[1])
	test_graph_obj.write_to_file(arguments[2])
	print test_graph_obj.attributes.items()[:10]
	print test_graph_obj.pgraph.GetNodes()
	print test_graph_obj.pgraph.GetEdges()
	print 'Done!'

def test_feature_extraction(arguments):
	print 'Extracting training examples.'
	train_graph_obj = Train_Graph(graph_file_root='smallest_train')
	train_pgraph = train_graph_obj.pgraph
	train_examples, train_labels = extract_examples(train_pgraph, 10000, 10000)

	print 'Extracting testing examples'
	test_graph_obj = Train_Graph(graph_file_root='smallest_test')
	test_pgraph = test_graph_obj.pgraph
	test_examples, test_labels = extract_test_examples(train_pgraph, test_pgraph, \
														train_examples, 1000, 1000)
	print 'Done!'

	print 'Graph Distance:', get_graph_distance(train_pgraph, train_examples[0][0], train_examples[0][1])
	print 'Common Neighbors:', get_common_neighbors(train_pgraph, train_examples[0][0], train_examples[0][1])
	print 'Jaccard Coefficient:', jaccard_coefficient(train_pgraph, train_examples[0][0], train_examples[0][1])
	print 'Adamic Adar:', adamic_adar(train_pgraph, train_examples[0][0], train_examples[0][1])
	print 'Preferential Attachment:', preferential_attachment(train_pgraph, train_examples[0][0], train_examples[0][1])


if __name__ == '__main__':
	arguments = sys.argv[1:]
	#test_graph_creation(arguments)
	# test_graph_loading(arguments)
	test_testing_graph_creation(arguments)
	# test_feature_extraction(arguments)
