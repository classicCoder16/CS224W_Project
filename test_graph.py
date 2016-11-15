import snap
import datetime
import sys
import numpy as np 

class Test_Graph:


	def __init__(self, year_lbound, year_ubound, node_file_root, src_path, graph_file_root=None):
		self.pgraph = snap.TUNGraph.New()
		# Maps the counter value to node attributes
		self.attributes = {}
		# Path to read the files from
		self.src_path = src_path
		# Years to filter on
		self.year_ubound = year_ubound
		self.year_lbound = year_lbound

		# If the graph file is specified, read from file
		if graph_file is not None: self.read_from_file(graph_file_root)

		# Else, create from scratch
		else: self.read_in_graph(node_file_root)

	def read_from_file(self, input_file_root, is_training=False):
		print 'Reading from file...'
		f = open(input_file_root + '_graph.txt')
		cutoffs = []
		for i in range(3):
			line = f.readline()
			cutoffs.append(tuple(map(int, line.split())))
		self.user_node_ids, self.board_node_ids, self.pin_node_ids = cutoffs
		for line in f:
			src_id, dst_id = map(int, line.split())
			if not self.pgraph.IsNode(src_id): self.pgraph.AddNode(src_id)
			if not self.pgraph.IsNode(dst_id): self.pgraph.AddNode(dst_id)
			if not is_training: self.pgraph.AddEdge(src_id, dst_id)
		f.close()

		print self.pgraph.GetNodes()
		print self.pgraph.GetEdges()
		print 'Reading attributes...'
		self.attributes = np.load(input_file_root + '_attr.npy').item()


	def write_to_file(self, output_file_root):
		print 'Saving to file...'
		f = open(output_file_root + '_graph.txt', 'w')
		cutoffs = [self.user_node_ids, self.board_node_ids, self.pin_node_ids]
		for cutoff in cutoffs:
			line = str(cutoff[0]) + '\t' + str(cutoff[1]) + '\n'
			f.write(line)
		for edge in self.pgraph.Edges():
			src_id = edge.GetSrcNId()
			dst_id = edge.GetDstNId()
			line = str(src_id) + '\t' + str(dst_id) + '\n'
			f.write(line)
		f.close()

		print 'Saving attributes to file...'
		np.save(self.attributes, output_file_root + '_attr.npy')


	def get_graph(self):
		return self.pgraph

	def read_in_graph(self, node_file_root):
		# Read the node ids from the saved
		# training graph
		self.read_from_file(node_file_root, is_training=True)
		# Remove all attribute values for old edges
		self.prune_attr()
		# Read new edges formed
		self.read_follows()
		self.read_pins()

	def prune_attr(self):
		for key in self.attributes:
			if isinstance(key, tuple):
				self.attributes.pop(key, None)


	def read_follows(self):
		print "Reading follows..."
		f = open(self.src_path + "follow.tsv")
		for line in f:
			follow_info = line.split('\t')
			# Break line into components
			board_id, user_id, time = follow_info
			board_id = self.get_mapped_board_id(int(board_id))
			user_id = int(user_id)

			# Year is first four characters of time
			follow_year = int(time[0:4])
			# Ignore invalid years
			if follow_year < self.year_lbound or follow_year > self.year_ubound: continue
			
			# Ignore current edge if neither node is in the training set
			if not self.pgraph.IsNode(user_id) or not self.pgrapg.IsNode(board_id):
				continue

			ret_val  = self.pgraph.AddEdge(user_id, board_id)
			self.attributes[(board_id, user_id)] = {'follow_time': time}
		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'


	def read_pins(self):
		print "Reading pins..."
		f = open(self.src_path + "pins.tsv")
		# For every line in file
		for line in f:

			# Split into attributes
			pins_info = line.split('\t')
			time, board_id, pin_id = pins_info
			pin_id = get_mapped_pin_id(int(pin_id))
			board_id = get_mapped_board_id(int(board_id))

			# Get year from unix timestamp
			pin_year = datetime.datetime.fromtimestamp(int(time)).year
			# Ignore pins outside of valid range
			if pin_year < self.year_lbound or pin_year > self.year_ubound: continue

			if not self.pgraph.IsNode(pin_id) or \
			 not self.pgraph.IsNode(board_id): continue
			
			# Add the new edge to the graph.
			self.pgraph.AddEdge(pin_id, board_id)
			self.attributes[(board_id, pin_id)] = {'pin_time': time}
		f.close()

		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'

	def get_mapped_board_id(self, board_id):
		return int(board_id) + self.user_node_ids[1] + 1

	def get_mapped_pin_id(self, pin_id):
		return int(pin_id) + self.board_node_ids[1] + 1

if __name__ == '__main__':
	src_path = sys.argv[1]
	pgraph_obj = Test_Graph(2014, 2020, 'train_graph', src_path)
	pgraph = pgraph_obj.get_graph()
	pgraph_obj.write_to_file('test')
	print 'Done!'
	print str(pgraph.GetNodes()) + ' Nodes'
	print str(pgraph.GetEdges()) + ' Edges'





