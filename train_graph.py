import snap
import datetime
import sys
import pickle 

class Train_Graph:


	def __init__(self, year_lbound, year_ubound, src_path, graph_file=None):
		self.pgraph = snap.TUNGraph.New()
		# Maps given board/user/pin ids to a counter value
		self.mapping = {}
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
		print "reading users"
		f = open(self.src_path + "users.tsv")
		# For every line
		for i, line in enumerate(f):
			user_id = 'u'+line.split()[0]
			# Add 1 since i starts at 0
			self.pgraph.AddNode(i+1)
			# Map the given user id to this counter
			self.mapping[user_id] = i+1
		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'
		self.user_node_ids = (1, self.pgraph.GetNodes())

	def read_boards(self):
		print "reading boards"
		f = open(self.src_path + "boards.tsv")
		# Start adding node ids beginning with
		# this index
		index = self.user_node_ids[1]
		for i, line in enumerate(f):

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

			# Ignore lines with invalid years
			board_year = datetime.datetime.fromtimestamp(int(time)).year
			if board_year < self.year_lbound or board_year > self.year_ubound: continue

			new_board_id = index + (i + 1)
			new_user_id = self.mapping['u' + user_id]

			# Map the node id to its attributes
			self.attributes[new_board_id] = {'name': board_name, 'description': description}
			# Add attribute for 'create' edge
			self.attributes[(new_board_id, new_user_id)] = {'create_time': time}

			# Add edge to user that created the board
			self.pgraph.AddNode(new_board_id)
			self.pgraph.AddEdge(new_board_id, new_user_id)

			# Map given board id to our counter id
			self.mapping['b'+board_id] = new_board_id
		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'
		self.board_node_ids = (self.user_node_ids[1] + 1, self.user_node_ids[1] + (i + 1))


	def read_follows(self):
		print "reading follows"
		f = open(self.src_path + "follow.tsv")
		for i, line in enumerate(f):
			follow_info = line.split('\t')
			# Break line into components
			board_id, user_id, time = follow_info
			# Year is first four characters of time
			follow_year = int(time[0:4])
			# Ignore invalid years
			if follow_year < self.year_lbound or follow_year > self.year_ubound: continue
			mapped_user_id = self.mapping['u'+user_id]
			mapped_board_id = self.mapping['b'+board_id]
			# Try adding edge; if exists, don't add attribute
			ret_val  = self.pgraph.AddEdge(mapped_user_id, mapped_board_id)
			if ret_val != -2:
				self.attributes[(mapped_board_id, mapped_user_id)] = {'follow_time': time}
			# Add edge from user to the board s/he follows
		f.close()
		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'


	def read_pins(self):
		print "reading pins"
		f = open(self.src_path + "pins.tsv")
		index = self.board_node_ids[1]
		counter = 0
		seen_before = set()
		# For every line in file
		for line in f:

			# Split into attributes
			pins_info = line.split('\t')
			time, board_id, pin_id = pins_info
			pin_id = str(int(pin_id))
			if pin_id not in seen_before:
				seen_before.add(pin_id)
				counter += 1
			# Get year from unix timestamp
			pin_year = datetime.datetime.fromtimestamp(int(time)).year
			# Ignore pins outside of valid range
			if pin_year < self.year_lbound or pin_year > self.year_ubound: continue
			# Ignore if board does not exist
			if'b'+board_id not in self.mapping: continue
			pid = index + counter
			bid = self.mapping['b'+board_id]
			# Add the pin node to the graph.
			if not self.pgraph.IsNode(pid): self.pgraph.AddNode(pid)
			self.pgraph.AddEdge(pid, bid)
			self.attributes[(bid, pid)] = {'pin_time': time}
		f.close()

		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'
		self.pin_node_ids = (self.board_node_ids[1] + 1, self.board_node_ids[1] + counter)


if __name__ == '__main__':
	src_path = sys.argv[1]
	pgraph_obj = Train_Graph(2010, 2013, src_path)
	pgraph = pgraph_obj.get_graph()
	pgraph_obj.write_to_file('train')
	print 'Done!'
	print str(pgraph.GetNodes()) + ' Nodes'
	print str(pgraph.GetEdges()) + ' Edges'





