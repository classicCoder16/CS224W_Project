import snap
import datetime

def read_in_graph(year_lbound, year_ubound):
	pgraph = snap.TUNGraph.New()
	mapping = {}
	attributes = {}
	read_users(pgraph, mapping)
	read_boards(pgraph, mapping, attributes, year_lbound, year_ubound)
	read_follows(pgraph, mapping, attributes, year_lbound, year_ubound)
	#read_pins(pgraph, mapping, year_lbound, year_ubound)
	return attributes, pgraph

def read_users(pgraph, mapping):
	print "reading users"
	f = open("..\\data\\users.tsv")
	for i, line in enumerate(f):
		user_id = 'u'+line.split()[0]
		pgraph.AddNode(i+1)
		mapping[user_id] = i+1
	f.close()

def read_boards(pgraph, mapping, attributes, year_lbound, year_ubound):
	print "reading boards"
	try:
		f = open("..\\data\\boards.tsv")
		index = len(mapping)
		for i, line in enumerate(f):
			board_info = line.split('\t')
			if len(board_info) < 4:
				board_id, user_id, time = board_info
				description = ""
				board_name = ""
			elif len(board_info) < 5:
				board_id, board_name, user_id, time = board_info
				description = ""
			else:
				board_id, board_name, description, user_id, time = board_info

			board_year = datetime.datetime.fromtimestamp(int(time)).year
			if board_year < year_lbound or board_year > year_ubound: continue

			attributes[index + i+ 1] = {'name': board_name, 'description': description, 'create_time': time}
			pgraph.AddNode(index + i + 1)
			if 'u'+user_id not in mapping: print 'u'+user_id
			pgraph.AddEdge(mapping['u'+user_id], index + i + 1)
			mapping['b'+board_id] = index + i + 1
		f.close()
	except Exception as e:
		print str(e)
		print board_info

def read_follows(pgraph, mapping, attributes, year_lbound, year_ubound):
	print "reading follows"
	f = open("..\\data\\follow.tsv")
	index = len(mapping)
	for i, line in enumerate(f):
		follow_info = line.split('\t')
		board_id, user_id, time = follow_info

		follow_year = int(time[0:4])
		if follow_year < year_lbound or follow_year > year_ubound: continue

		pgraph.AddEdge(mapping['u'+user_id], mapping['b'+board_id])
	f.close()

def read_pins(pgraph, mapping, attributes, year_lbound, year_ubound):
	print "reading pins"
	f = open("..\\data\\pins.tsv")
	index = len(mapping)
	counter = 1
	for line in f:
		pins_info = line.split('\t')
		board_id, time, pin_id = pins_info
		pin_year = datetime.datetime.fromtimestamp(int(time)).year
		if pin_year < year_lbound or pin_year > year_ubound: continue
		if 'p' + pin_id not in mapping:
			mapping['p'+pin_id] = index + counter
			counter += 1
			pgraph.AddNode(index+counter)
		pgraph.AddEdge(index+counter, mapping['b'+board_id])
	f.close()



attributes, pgraph = read_in_graph(2010, 2013)





