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

def main(arguments):
	print 'Testing train_graph.py...'
	time_lbound = datetime.datetime(2013, 1, 1)
	time_ubound = datetime.datetime(2013, 2, 28)
	train_graph_obj = Train_Graph(time_lbound=time_lbound, \
								  time_ubound=time_ubound, \
								  src_path=arguments[0])
	print train_graph_obj.attrbutes.items()[:10]
	print train_graph_obj.pgraph.GetNodes()
	print train_graph_obj.pgraph.GetEdges()
	print 'Done!'

	train_graph_obj.write_to_file()



if __name__ == '__main__':
	main(sys.argv[1:])
