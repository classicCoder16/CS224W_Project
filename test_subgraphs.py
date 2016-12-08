import snap

def get_test_graph(train_name, test_name, train_prime_name, test_prune_name, g_type):
	train = snap.LoadEdgeList(snap.PUNGraph, train_name, 0, 1)
	train.DelNode(0)
	train.DelEdge(18630354, 31097108)
	train.DelEdge(31097109, 51147066)
	print "read training graph"
	print "reading test graph"
	f = open(test_name, "r")
	t1 = f.readline()
	t2 = f.readline()
	t3 = f.readline()
	for line in f: 
		e1, e2 = line.split()
		if train.IsNode(int(e1)) and train.IsNode(int(e2)):	train.AddEdge(int(e1), int(e2))
	f.close()
	print "read test graph"
	print "prune test graph"
	pruned = process_graph(train, g_type)
	f = open(train_prime_name, "r")
	for line in f:
		e1, e2 = line.split()
		pruned.DelEdge(int(e1), int(e2))
	f.close()
	snap.SaveEdgeList(pruned, test_prune_name)

def process_graph(G, g_type):
	if g_type == 'u':
		return get_users_unweighted(G)
	elif g_type == 'p':
		return get_pins_unweighted(G)
	elif g_type == 'bu':
		return get_board_users_unweighted(G)
	elif g_type == 'bp':
		return get_board_pins_unweighted(G)
	else:
		return get_board_unweighted(G)


def get_users_unweighted(G):
	print "reading users"
	new_G = snap.PUNGraph.New()
	for n in G.Nodes():
		n = n.GetId()
		if n > 31097108: G.DelNode(n)
	nodes = sorted([n.GetId() for n in G.Nodes()])
	count = 0
	for n in nodes:
		if n > 18630353: break
		if not new_G.IsNode(n): new_G.AddNode(n)
		users = snap.TIntV()
		snap.GetNodesAtHop(G, n, 2, users, False)
		for u in users:
			if not new_G.IsNode(u): new_G.AddNode(u)
			new_G.AddEdge(n, u)
		G.DelNode(n)
		if count%10000 == 0: print count
		count += 1
	print "users read"
	return new_G

	

def get_pins_unweighted(G):
	print "reading pins"
	new_G = snap.PUNGraph.New()
	for n in G.Nodes():
		n = n.GetId()
		if n <= 18630353: G.DelNode(n)
	nodes = sorted([n.GetId() for n in G.Nodes()], reverse=True)
	count = 0
	for n in nodes:
		if n <= 31097108: break
		if not new_G.IsNode(n): new_G.AddNode(n)
		pins = snap.TIntV()
		snap.GetNodesAtHop(G, n, 2, pins, False)
		for p in pins:
			if not new_G.IsNode(p): new_G.AddNode(p)
			new_G.AddEdge(n, p)
		G.DelNode(n)
		if count%10000 == 0: print count
		count += 1
	print "pins read"
	return new_G

	

def get_board_users_unweighted(G):
	print "reading boards"
	new_G = snap.PUNGraph.New()
	for n in G.Nodes():
		n = n.GetId()
		if n > 31097108: G.DelNode(n)
	nodes = sorted([n.GetId() for n in G.Nodes()], reverse=True)
	count = 0
	for n in nodes:
		if n <= 18630353: break
		if not new_G.IsNode(n): new_G.AddNode(n)
		boards = snap.TIntV()
		snap.GetNodesAtHop(G, n, 2, boards, False)
		for b in boards:
			if not new_G.IsNode(b): new_G.AddNode(b)
			new_G.AddEdge(n, b)
		G.DelNode(n)
		if count%10000 == 0: print count
		count += 1
	print "boards read"
	return new_G

def get_board_pins_unweighted(G):
	print "reading boards"
	new_G = snap.PUNGraph.New()
	for n in G.Nodes():
		n = n.GetId()
		if n <= 18630353: G.DelNode(n)
	nodes = sorted([n.GetId() for n in G.Nodes()])
	count = 0
	for n in nodes:
		if n > 31097108: break
		if not new_G.IsNode(n): new_G.AddNode(n)
		boards = snap.TIntV()
		snap.GetNodesAtHop(G, n, 2, boards, False)
		for b in boards:
			if not new_G.IsNode(b): new_G.AddNode(b)
			new_G.AddEdge(n, b)
		G.DelNode(n)
		if count%10000 == 0: print count
		count += 1
	print "boards read"
	return new_G

def get_board_unweighted(G):
	print "reading boards"
	new_G = snap.PUNGraph.New()
	nodes = sorted([n.GetId() for n in G.Nodes()])
	count = 0
	for n in nodes:
		if n <= 18630353: continue
		if n > 31097108: break
		if not new_G.IsNode(n): new_G.AddNode(n)
		boards = snap.TIntV()
		snap.GetNodesAtHop(G, n, 2, boards, False)
		for b in boards:
			if b > 18630353 and b <= 31097108:
				if not new_G.IsNode(b): new_G.AddNode(b)
				new_G.AddEdge(n, b)
		G.DelNode(n)
		if count%10000 == 0: print count
		count += 1
	print "boards read"
	return new_G

def main():
	get_test_graph('smallest_train_graph.txt', "smallest_test_graph.txt", "user_small_train.txt", "user_small_test.txt",  "u")
	get_test_graph('smallest_train_graph.txt', "smallest_test_graph.txt", "pins_small_train.txt", "pins_small_test.txt",  'p')
	get_test_graph('smallest_train_graph.txt', "smallest_test_graph.txt", "user_board_small_train.txt", "user_board_small_test.txt",  'bu')
	get_test_graph('smallest_train_graph.txt', "smallest_test_graph.txt", "pin_board_small_train.txt", "pin_board_small_test.txt",  'bp')
	get_test_graph('smallest_train_graph.txt', "smallest_test_graph.txt", "board_train.txt", "board_test.txt",  'b')
main()