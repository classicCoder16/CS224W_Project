import snap
import datetime

def read_in_graph(year_lbound, year_ubound):
	pgraph = snap.TUNGraph.New()
	mapping = {}
	attributes = {}
	read_users(pgraph, mapping)
	read_boards(pgraph, mapping, attributes, year_lbound, year_ubound)
	read_follows(pgraph, mapping, year_lbound, year_ubound)
	read_pins(pgraph, mapping, year_lbound, year_ubound)

def read_users(pgraph, mapping):
	f = open("users.tsv")
	for i, line in enumerate(f):
		user_id = int(line)
		pgraph.AddNode(i+1)
		mapping[user_id] = i+1
	f.close()

def read_boards(pgraph, mapping, attributes, year_lbound, year_ubound):
	f = open("boards.tsv")
	index = len(mapping)
	for i, line in enumerate(f):
		board_info = line.split('\t')
		if len(board_info) < 5:
			board_id, board_name, user_id, time = board_info
			description = ""
		else:
			board_id, board_name, description, user_id, time = board_info

		board_year = datetime.datetime.fromtimestamp(int(time)).year
		if board_year < year_lbound or board_year > year_ubound: continue

		attributes[index + i+ 1] = {'name': board_name, 'description': description, 'create_time': time}
		pgraph.AddNode(index + i + 1)
		pgraph.AddEdge(mapping[int(user_id)], index + i + 1)
		mapping[int(board_id)] = index + i + 1
	f.close()

def read_follows(pgraph, mapping, attributes, year_lbound, year_ubound):
	f = open("follows.tsv")
	index = len(mapping)
	for i, line in enumerate(f):
		follow_info = line.split('\t')
		board_id, user_id, time = follow_info

		follow_year = datetime.datetime.fromtimestamp(int(time)).year
		if follow_year < year_lbound or follow_year > year_ubound: continue

		pgraph.AddEdge(mapping[int(user_id)], mapping[int(board_id)])
	f.close()








