import snap
import datetime
import sys
import pickle 

class Test_Graph:


	def __init__(self, year_lbound, year_ubound, node_file_root, src_path, graph_file_root=None):
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
		if graph_file is not None: self.read_from_file(graph_file_root)

		# Else, create from scratch
		else: self.read_in_graph(node_file_root)

	def read_from_file(self, input_file_root, is_training=False):
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
			if not is_training: self.pgraph.AddEdge(src_id, dst_id)
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

	def read_users(self):
		print "reading users"
		f = open(self.src_path + "users.tsv")
		# For every line
		for i, line in enumerate(f):
			user_id = 'u'+line.split()[0]
			# Map the given user id to this counter
			self.mapping[user_id] = i+1
		f.close()

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
			board_id = board_info[0]

			new_board_id = index + (i + 1)
			if not self.pgraph.IsNode(new_board_id): continue

			# Map given board id to our counter id
			self.mapping['b'+board_id] = new_board_id
		f.close()

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
			# Ignore current edge if neither node is in the training set
			if not self.pgraph.IsNode(mapped_user_id) or not self.pgrapg.IsNode(mapped_board_id):
				continue

			# Try adding edge; if exists, don't add attribute
			ret_val  = self.pgraph.AddEdge(mapped_user_id, mapped_board_id)
			self.attributes[(mapped_board_id, mapped_user_id)] = {'follow_time': time}
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
			# Advance counter if a new pin id
			if pin_id not in seen_before:
				seen_before.add(pin_id)
				counter += 1
			# Get year from unix timestamp
			pin_year = datetime.datetime.fromtimestamp(int(time)).year
			# Ignore pins outside of valid range
			if pin_year < self.year_lbound or pin_year > self.year_ubound: continue
			# Ignore if pin or board are not existing nodes
			pid = index + counter
			if not self.pgraph.IsNode(pid) or 'b'+board_id not in self.mapping: continue
			
			bid = self.mapping['b'+board_id]
			# Add the new edge to the graph.
			self.pgraph.AddEdge(pid, bid)
			self.attributes[(bid, pid)] = {'pin_time': time}
		f.close()

		print str(self.pgraph.GetNodes()) + ' Nodes'
		print str(self.pgraph.GetEdges()) + ' Edges'

if __name__ == '__main__':
	src_path = sys.argv[1]
	pgraph_obj = Test_Graph(2014, 2020, 'train_graph', src_path)
	pgraph = pgraph_obj.get_graph()
	pgraph_obj.write_to_file('test')
	print 'Done!'
	print str(pgraph.GetNodes()) + ' Nodes'
	print str(pgraph.GetEdges()) + ' Edges'





