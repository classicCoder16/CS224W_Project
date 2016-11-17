import snap

def get_users_unweighted(G):
	print "reading users"
	f = open("user_small_test.txt", "w")
	for n in G.Nodes():
		n = n.GetId()
		if n > 31097108: G.DelNode(n)
	nodes = sorted([n.GetId() for n in G.Nodes()])
	u_edges = {}
	count = 0
	for n in nodes:
		if n > 18630353: break
		users = snap.TIntV()
		snap.GetNodesAtHop(G, n, 2, users, False)
		for u in users:
			f.write(str(n) + "\t" + str(u) + '\n')
		G.DelNode(n)
		if count%10000 == 0: print count
		count += 1
	f.close()

	print "users read"

def get_pins_unweighted(G):
	print "reading pins"
	f = open("pins_small_test.txt", "w")
	for n in G.Nodes():
		n = n.GetId()
		if n <= 18630353: G.DelNode(n)
	nodes = sorted([n.GetId() for n in G.Nodes()], reverse=True)
	u_edges = {}
	count = 0
	for n in nodes:
		if n <= 31097108: break
		pins = snap.TIntV()
		snap.GetNodesAtHop(G, n, 2, pins, False)
		for p in pins:
			f.write(str(n) + "\t" + str(p) + '\n')
		G.DelNode(n)
		if count%10000 == 0: print count
		count += 1
	f.close()

	print "pins read"

def get_board_users_unweighted(G):
	print "reading boards"
	f = open("user_board_small_test.txt", "w")
	for n in G.Nodes():
		n = n.GetId()
		if n > 31097108: G.DelNode(n)
	nodes = sorted([n.GetId() for n in G.Nodes()], reverse=True)
	count = 0
	for n in nodes:
		if n <= 18630353: break
		boards = snap.TIntV()
		snap.GetNodesAtHop(G, n, 2, boards, False)
		for b in boards:
			f.write(str(n) + "\t" + str(b) + '\n')
		G.DelNode(n)
		if count%10000 == 0: print count
		count += 1
	f.close()

def get_board_pins_unweighted(G):
	print "reading boards"
	f = open("pin_board_small_test.txt", "w")
	for n in G.Nodes():
		n = n.GetId()
		if n <= 18630353: G.DelNode(n)
	nodes = sorted([n.GetId() for n in G.Nodes()])
	count = 0
	for n in nodes:
		if n > 31097108: break
		boards = snap.TIntV()
		snap.GetNodesAtHop(G, n, 2, boards, False)
		for b in boards:
			f.write(str(n) + "\t" + str(b) + '\n')
		G.DelNode(n)
		if count%10000 == 0: print count
		count += 1
	f.close()

def get_board_unweighted(G):
	print "reading boards"
	f = open("board_small_test.txt", "w")
	nodes = sorted([n.GetId() for n in G.Nodes()])
	count = 0
	for n in nodes:
		if n <= 18630353: continue
		if n > 31097108: break
		boards = snap.TIntV()
		snap.GetNodesAtHop(G, n, 2, boards, False)
		for b in boards:
			if b > 18630353 and b <= 31097108: f.write(str(n) + "\t" + str(b) + '\n')
		G.DelNode(n)
		if count%10000 == 0: print count
		count += 1
	f.close()

def main():
	G = snap.LoadEdgeList(snap.PUNGraph, "smallest_test_graph.txt", 0, 1)
	G.DelNode(0)
	get_users_unweighted(G)
	G = snap.LoadEdgeList(snap.PUNGraph, "smallest_test_graph.txt", 0, 1)
	G.DelNode(0)
	get_board_users_unweighted(G)
	G = snap.LoadEdgeList(snap.PUNGraph, "smallest_test_graph.txt", 0, 1)
	G.DelNode(0)
	get_pins_unweighted(G)
	G = snap.LoadEdgeList(snap.PUNGraph, "smallest_test_graph.txt", 0, 1)
	G.DelNode(0)
	get_board_pins_unweighted(G)
	G = snap.LoadEdgeList(snap.PUNGraph, "smallest_test_graph.txt", 0, 1)
	G.DelNode(0)
	get_board_unweighted(G)

main()