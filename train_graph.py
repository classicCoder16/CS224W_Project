import snap
import datetime
import sys
import numpy as np
import time
class Train_Graph:

	'''
	Function: __init__
	------------------
	Initializes the state of the training graph. 
	Reads in the graph from scratch from the original
	data files, or reads in an existing graph if
	graph file is passed in. 
	'''
	def __init__(self, time_lbound=None, \
				time_ubound=None, src_path=None, \
				graph_file_root=None):

		self.pgraph = snap.TUNGraph.New()

		# Maps the unique node-id to node attributes
		self.attributes = {}

		# Path to read the files from
		self.src_path = src_path

		# Times to filter on, in the form of datetimes.
		# You can create a datetime object by
		# dt = datetime.datetime(2012, 10, 31)
		self.time_lbound = time_lbound
		self.time_ubound = time_ubound

		# If the graph file is specified, read from file
		if graph_file_root is not None: self.read_from_file(graph_file_root)

		# Else, create from scratch
		else: self.read_in_graph()


	'''
	Function: read_from_file
	------------------------
	Reads an existing graph and its attributes from the 
	provided filename root.
	'''
	def read_from_file(self, input_file_root):
		print 'Reading from existing file...'
		f = open(input_file_root + '_graph.txt')
		cutoffs = []

		# Read three lines to get the cutoffs
		for i in range(3):
			line = f.readline()
			cutoffs.append(tuple(map(int, line.split())))
		self.user_node_ids, self.board_node_ids, self.pin_node_ids = cutoffs

		# Read the rest of the lines to construct the graph
		for line in f:
			src_id, dst_id = map(int, line.split())
			if not self.pgraph.IsNode(src_id): self.pgraph.AddNode(src_id)
			if not self.pgraph.IsNode(dst_id): self.pgraph.AddNode(dst_id)
			self.pgraph.AddEdge(src_id, dst_id)

		f.close()
		print self.pgraph.GetNodes()
		print self.pgraph.GetEdges()

		# Read the attributes associated with the graph
		print 'Reading attributes...'
		self.attributes = np.load(input_file_root + '_attr.npy').item()


	'''
	Function: write_to_file
	------------------------
	Reads an existing graph and its attributes from the 
	provided filename root.
	'''
	def write_to_file(self, output_file_root):
		print 'Writing to file'
		f = open(output_file_root + '_graph.txt', 'w')

		# Make the first three lines of the file the cutoffs
		cutoffs = [self.user_node_ids, self.board_node_ids, self.pin_node_ids]
		for cutoff in cutoffs:
			line = str(cutoff[0]) + '\t' + str(cutoff[1]) + '\n'
			f.write(line)

		# Write out each edge of the graph
		for edge in self.pgraph.Edges():
			src_id = edge.GetSrcNId()
			dst_id = edge.GetDstNId()
			line = str(src_id) + '\t' + str(dst_id) + '\n'
			f.write(line)

		f.close()

		# Save the attributes with numpy
		np.save(output_file_root + '_attr.npy', self.attributes)


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
	Reads in the graph from the four
	data files.
	'''
	def read_in_graph(self):
		# Read all the data from the four files
		self.read_users()
		self.read_boards()
		self.read_follows()
		self.read_pins()


	'''
	Function: read_users
	-----------------------
	Reads in the user nodes from the
	user file.
	'''
	def read_users(self):
		print "Reading users..."
		f = open(self.src_path + "users.tsv")
		max_user_id = 0

		# For every line
		for line in f:

			# Get the user id, add to graph
			user_id = int(line.split()[0])
			if user_id > max_user_id: max_user_id = user_id

		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'
		self.user_node_ids = (0, max_user_id)
		print 'Range:', self.user_node_ids


	'''
	Function: read_boards
	-----------------------
	Reads in all board nodes
	that were created in the given span of time.
	'''
	def read_boards(self):
		print "Reading boards..."
		f = open(self.src_path + "boards.tsv")

		max_board_id = 0
		for line in f:

			# Split by tab since description/name 
			# may have spaces
			board_info = line.split('\t')

			# If description not present
			if len(board_info) < 5:
				board_id, board_name, user_id, board_time = board_info
				description = ""
			# Else all attributes are present
			else:
				board_id, board_name, description, user_id, old_board_time = board_info

			# Map given board id to a unique number
			board_id = self.get_mapped_board_id(int(board_id))
			if board_id > max_board_id: max_board_id = board_id

			# Ignore lines with invalid times
			board_time = datetime.datetime.fromtimestamp(int(old_board_time))
			if board_time < self.time_lbound or board_time > self.time_ubound: continue

			# Add the user that created the board
			user_id = int(user_id)
			if not self.pgraph.IsNode(user_id): self.pgraph.AddNode(user_id)

			# Map the node id to its attributes
			self.attributes[board_id] = {'name': board_name, 'description': description}
			
			# Add attribute for 'created' edge
			self.attributes[(board_id, user_id)] = {'create_time': int(old_board_time)}

			# Add edge to user that created the board
			self.pgraph.AddNode(board_id)
			self.pgraph.AddEdge(board_id, user_id)

		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'

		# Set cutoff for board ids
		self.board_node_ids = (1 + self.user_node_ids[1], max_board_id)
		print 'Range:', self.board_node_ids


	'''
	Function: read_follows
	-----------------------
	Reads in all 'follows' edges
	that were formed in the given span of time
	over existing nodes.
	'''
	def read_follows(self):
		print "Reading follows..."
		f = open(self.src_path + "follow.tsv")
		for line in f:

			# Break line into components
			follow_info = line.split('\t')
			board_id, user_id, follow_time = follow_info

			# Get the board and user id involved
			board_id = self.get_mapped_board_id(int(board_id))
			user_id = int(user_id)

			# Get the datetime object from the string
			follow_time = datetime.datetime.strptime(follow_time.split()[0], '%Y-%m-%d')

			# Ignore invalid times
			if follow_time < self.time_lbound or follow_time > self.time_ubound: continue

			# Ignore line if board node is present
			if not self.pgraph.IsNode(board_id): continue

			# Add the user to the graph
			if not self.pgraph.IsNode(user_id): self.pgraph.AddNode(user_id)

			# Try adding edge; if exists, don't add attribute
			ret_val  = self.pgraph.AddEdge(user_id, board_id)
			if ret_val != -2: 
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
		max_pin_id = 0

		counter = 0
		# For every line in file
		for line in f:

			# Print every million lines
			counter += 1
			if (counter % 1000000) == 0: print 'Line', counter

			# Split into attributes
			pins_info = line.split('\t')
			old_pin_time, board_id, pin_id = pins_info

			# Get the pin and board involved
			pin_id = self.get_mapped_pin_id(int(pin_id))
			board_id = self.get_mapped_board_id(int(board_id))

			if pin_id > max_pin_id: max_pin_id = pin_id

			# Get time from unix timestamp
			pin_time = datetime.datetime.fromtimestamp(int(old_pin_time))

			# Ignore pins outside of valid time range
			if pin_time < self.time_lbound or pin_time > self.time_ubound: continue
			
			# Don't add an edge if the board doesn't exist
			if not self.pgraph.IsNode(board_id): continue

			# Add the pin node to the graph.
			if not self.pgraph.IsNode(pin_id): self.pgraph.AddNode(pin_id)
			
			# Add an edge between the pin and the board			
			self.pgraph.AddEdge(pin_id, board_id)

			# Keep track of edge attributes
			self.attributes[(board_id, pin_id)] = {'pin_time': int(old_pin_time)}

		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'
		self.pin_node_ids = (self.board_node_ids[1] + 1, max_pin_id)
		print 'Range:', self.pin_node_ids


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






