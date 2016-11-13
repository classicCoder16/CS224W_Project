import snap
import datetime
import sys
import pickle 

class Train_Graph:


	def __init__(self, year_lbound, year_ubound, src_path, graph_file=None):
		self.pgraph = snap.TUNGraph.New()
		# Maps given board/user/pin ids to a counter value
		# self.mapping = {}
		# Maps the counter value to node attributes
		self.attributes = {}
		# Path to read the files from
		self.src_path = src_path
		# Years to filter on
		self.year_ubound = year_ubound
		self.year_lbound = year_lbound

		# If the graph file is specified, read from file
		if graph_file is not None: self.read_from_file(graph_file)

		# Else, create from scratch
		else: self.read_in_graph()

	def read_from_file(self, input_file_root):
		f = open(output_file_root + '_graph.txt')
		cutoffs = []
		for i in range(3):
			line = f.readline()
			cutoffs.append(tuple(map(int, line.split())))
		self.user_node_ids, self.board_node_ids, self.pin_node_ids = cutoffs
		for line in f:
			src_id, dst_id = map(int, line.split())
			if not self.pgraph.IsNode(src_id): self.pgraph.AddNode(src_id)
			if not self.pgraph.IsNode(dst_id): self.pgraph.AddNode(dst_id)
			self.pgraph.AddEdge(src_id, dst_id)
		f.close()

		g = open(output_file_root + '_attr.pickle', 'rb')
		self.attributes = pickle.load(g)
		g.close()

	def write_to_file(self, output_file_root):
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

		g = open(output_file_root + '_attr.pickle', 'wb')
		pickle.dump(self.attributes, g)
		g.close()


	def get_graph(self):
		return self.pgraph

	def read_in_graph(self):
		# Read all the data from the four files
		self.read_users()
		self.read_boards()
		self.read_follows()
		self.read_pins()

	def read_users(self):
		print "Reading users..."
		f = open(self.src_path + "users.tsv")
		# For every line
		max_user_id = 0
		for line in f:
			user_id = int(line.split()[0])
			if user_id > max_user_id: max_user_id = user_id
			self.pgraph.AddNode(user_id)
		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'
		self.user_node_ids = (0, max_user_id)
		print 'Range:', self.user_node_ids

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
				board_id, board_name, user_id, time = board_info
				description = ""
			# Else all attributes are present
			else:
				board_id, board_name, description, user_id, time = board_info

			board_id = self.get_mapped_board_id(int(board_id))
			if board_id > max_board_id: max_board_id = board_id

			# Ignore lines with invalid years
			board_year = datetime.datetime.fromtimestamp(int(time)).year
			if board_year < self.year_lbound or board_year > self.year_ubound: continue

			user_id = int(user_id)

			# Map the node id to its attributes
			self.attributes[board_id] = {'name': board_name, 'description': description}
			# Add attribute for 'create' edge
			self.attributes[(board_id, user_id)] = {'create_time': time}

			# Add edge to user that created the board
			self.pgraph.AddNode(board_id)
			self.pgraph.AddEdge(board_id, user_id)

		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'
		self.board_node_ids = (1 + self.user_node_ids[1], max_board_id)
		print 'Range:', self.board_node_ids


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

			if not self.pgraph.IsNode(user_id) or not self.pgraph.IsNode(board_id): continue

			# Try adding edge; if exists, don't add attribute
			ret_val  = self.pgraph.AddEdge(user_id, board_id)
			if ret_val != -2:
				self.attributes[(board_id, user_id)] = {'follow_time': time}
		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'


	def read_pins(self):
		print "Reading pins..."
		f = open(self.src_path + "pins.tsv")
		# For every line in file
		max_pin_id = 0
		for line in f:

			# Split into attributes
			pins_info = line.split('\t')
			time, board_id, pin_id = pins_info

			pin_id = self.get_mapped_pin_id(int(pin_id))
			board_id = self.get_mapped_board_id(int(board_id))

			if pin_id > max_pin_id: max_pin_id = pin_id

			# Get year from unix timestamp
			pin_year = datetime.datetime.fromtimestamp(int(time)).year

			# Ignore pins outside of valid range
			if pin_year < self.year_lbound or pin_year > self.year_ubound: continue
			
			# Add the pin node to the graph.
			if not self.pgraph.IsNode(pin_id): self.pgraph.AddNode(pin_id)
			
			# Don't add an edge if the board doesn't exist
			if not self.pgraph.IsNode(board_id): continue

			self.pgraph.AddEdge(pin_id, board_id)
			self.attributes[(board_id, pin_id)] = {'pin_time': time}

		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'
		self.pin_node_ids = (self.board_node_ids[1] + 1, max_pin_id)
		print 'Range:', self.pin_node_ids

	def get_mapped_board_id(self, board_id):
		return int(board_id) + self.user_node_ids[1] + 1

	def get_mapped_pin_id(self, pin_id):
		return int(pin_id) + self.board_node_ids[1] + 1

if __name__ == '__main__':
	src_path = sys.argv[1]
	pgraph_obj = Train_Graph(2013, 2013, src_path)
	pgraph = pgraph_obj.get_graph()
	pgraph_obj.write_to_file(sys.argv[2])
	print 'Done!'
	print str(pgraph.GetNodes()) + ' Nodes'
	print str(pgraph.GetEdges()) + ' Edges'
	print self.user_node_ids
	print self.board_node_ids
	print self.pin_node_ids





