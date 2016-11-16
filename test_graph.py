import snap
import datetime
import sys
import numpy as np 

class Test_Graph:

	'''
	Function: __init__
	------------------
	Initializes the state of the test graph. 
	Reads in the graph from scratch based on a passed in
	training graph, or reads in an existing testing graph if
	graph file is passed in. 
	'''
	def __init__(self, time_lbound=None, \
				time_ubound=None, node_file_root=None, \
				src_path=None, graph_file_root=None):

		self.pgraph = snap.TUNGraph.New()

		# Maps the counter value to node attributes
		self.attributes = {}

		# Path to read the files from
		self.src_path = src_path

		# Times to filter on
		self.time_ubound = time_ubound
		self.time_lbound = time_lbound

		# If the graph file is specified, read from file
		if graph_file is not None: self.read_from_file(graph_file_root)

		# Else, create from scratch
		else: self.read_in_graph(node_file_root)


	'''
	Function: read_from_file
	------------------------
	Reads an existing graph and its attributes from the 
	provided filename root. If is_training is set to true,
	we read in only the node values, and don't add edges.
	'''
	def read_from_file(self, input_file_root, is_training=False):
		print 'Reading from file...'
		f = open(input_file_root + '_graph.txt')

		# Read in the first three lines as the cutoffs
		cutoffs = []
		for i in range(3):
			line = f.readline()
			cutoffs.append(tuple(map(int, line.split())))
		self.user_node_ids, self.board_node_ids, self.pin_node_ids = cutoffs

		# Read in every other line to get the graph
		# structure
		for line in f:
			src_id, dst_id = map(int, line.split())
			if not self.pgraph.IsNode(src_id): self.pgraph.AddNode(src_id)
			if not self.pgraph.IsNode(dst_id): self.pgraph.AddNode(dst_id)

			# Only add an edge if this is not the training graph
			# we're reading in
			if not is_training: self.pgraph.AddEdge(src_id, dst_id)
		f.close()

		print self.pgraph.GetNodes()
		print self.pgraph.GetEdges()
		print 'Reading attributes...'

		# Read in the attributes for the graph
		self.attributes = np.load(input_file_root + '_attr.npy').item()


	'''
	Function: write_to_file
	------------------------
	Writes an existing graph and its attributes to the 
	provided filename root.
	'''
	def write_to_file(self, output_file_root):
		print 'Saving to file...'
		f = open(output_file_root + '_graph.txt', 'w')

		# Write the cutoffs to the first three lines
		cutoffs = [self.user_node_ids, self.board_node_ids, self.pin_node_ids]
		for cutoff in cutoffs:
			line = str(cutoff[0]) + '\t' + str(cutoff[1]) + '\n'
			f.write(line)

		# Write the edges to the other lines.
		for edge in self.pgraph.Edges():
			src_id = edge.GetSrcNId()
			dst_id = edge.GetDstNId()
			line = str(src_id) + '\t' + str(dst_id) + '\n'
			f.write(line)
		f.close()

		print 'Saving attributes to file...'
		np.save(self.attributes, output_file_root + '_attr.npy')


	'''
	Function: get_graph
	-------------------
	Returns the snap graph object
	'''
	def get_graph(self):
		return self.pgraph


	'''
	Function: read_in_graph
	-----------------------
	Reads in the training graph,
	and then adds edges from the follows
	and pins files based on the existing nodes.
	'''
	def read_in_graph(self, node_file_root):
		# Read the node ids from the saved training graph
		self.read_from_file(node_file_root, is_training=True)

		# Remove all attribute values for old edges
		self.prune_attr()

		# Read new edges formed
		self.read_follows()
		self.read_pins()


	'''
	Function: prune_attr
	-----------------------
	Removes all attributes for edges in
	self.attributes.
	'''
	def prune_attr(self):
		for key in self.attributes:
			if isinstance(key, tuple):
				self.attributes.pop(key, None)

	'''
	Function: read_follows
	-----------------------
	Reads in all 'follows' edges that were 
	formed in the given span of time over existing nodes.
	'''
	def read_follows(self):
		print "Reading follows..."
		f = open(self.src_path + "follow.tsv")
		for line in f:
			# Break line into components
			follow_info = line.split('\t')
			board_id, user_id, follow_time = follow_info

			# Get the board and user id involved.
			board_id = self.get_mapped_board_id(int(board_id))
			user_id = int(user_id)

			# Get the datetime object from the string
			follow_time = datetime.datetime.strptime(follow_time, '%Y-%m-%d')

			# Ignore invalid times
			if follow_time < self.time_lbound or follow_time > self.time_ubound: continue
			
			# Ignore current edge if neither node is in the training set
			if not self.pgraph.IsNode(user_id) or not self.pgrapg.IsNode(board_id):
				continue

			# Add the edge, and set the attribute
			self.pgraph.AddEdge(user_id, board_id)
			follow_time = int(time.mktime(follow_time.timetuple()))
			self.attributes[(board_id, user_id)] = {'follow_time': follow_time}

		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'


	'''
	Function: read_pins
	-----------------------
	Reads in all pin nodes and their corresponding edges
	that were formed in the given span of time
	over existing nodes.
	'''
	def read_pins(self):
		print "Reading pins..."
		f = open(self.src_path + "pins.tsv")

		# For every line in file
		for line in f:

			# Split into attributes
			pins_info = line.split('\t')
			old_pin_time, board_id, pin_id = pins_info

			# Get the pin id and board id involved
			pin_id = get_mapped_pin_id(int(pin_id))
			board_id = get_mapped_board_id(int(board_id))

			# Get time from unix timestamp
			pin_time = datetime.datetime.fromtimestamp(int(old_pin_time))

			# Ignore pins outside of valid range
			if pin_time < self.time_lbound or pin_time > self.time_ubound: continue

			# Ignore if either the pin or board is not present in the training
			# graph
			if not self.pgraph.IsNode(pin_id) or \
			 not self.pgraph.IsNode(board_id): continue
			
			# Add the new edge to the graph.
			self.pgraph.AddEdge(pin_id, board_id)
			self.attributes[(board_id, pin_id)] = {'pin_time': int(old_pin_time)}

		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'


	'''
	Function: get_mapped_board_id
	-----------------------
	Helper function that maps a given board id
	to its unique identifier
	'''
	def get_mapped_board_id(self, board_id):
		return int(board_id) + self.user_node_ids[1] + 1


	'''
	Function: get_mapped_pin_id
	-----------------------
	Helper function that maps a given pin id
	to its unique identifier
	'''
	def get_mapped_pin_id(self, pin_id):
		return int(pin_id) + self.board_node_ids[1] + 1






