import collections


def main():
	user_board_list = {}
	pin_board_list = {}
	board_u_list = {}
	board_p_list = {}

	read_graph(user_board_list, pin_board_list, board_u_list, board_p_list)
	print "Read in Graph"
	get_user_subgraph(user_board_list, board_u_list)
	print "did users"
	get_pin_subgraph(pin_board_list)
	print "did pins"
	get_board_subgraphs(board_u_list, board_p_list)
	print "did boards"

def read_graph(user_board_list, pin_board_list, board_u_list, board_p_list):

	f = open("real_train_graph.txt", "r")
	u_range = f.readline().split()[1]
	b_range = f.readline().split()
	p_range = f.readline().split()[0]
	for line in f:
		n1, n2 = line.split()
		min_n = n1 if n1 < n2 else n2
		max_n = n1 if min_n == n2 else n2
		if min_n <= u_range:
			if min_n not in user_board_list:
				user_board_list[min_n] = [max_n]
			else:
				user_board_list[min_n].append(max_n)
			
			if max_n not in board_u_list:
				board_u_list[max_n] = [min_n]
			else:
				board_u_list[max_n].append(min_n)
			
		elif max_n >= p_range:
			if max_n not in pin_board_list:
				pin_board_list[max_n] = [min_n]
			else:
				pin_board_list[max_n].append(min_n)
			if min_n not in board_p_list:
				board_p_list[min_n] = [max_n]
			else:
				board_p_list[min_n].append(max_n)
	f.close()
	print len(user_board_list)
	print len(pin_board_list)
	print len(board_u_list)
	print len(board_p_list)

def get_user_subgraph(user_board_list, board_u_list):
	user_user = {}
	for u1 in user_board_list:
		for b in user_board_list[u1]:
			for u2 in board_u_list[b]:
				tup = (u1, u2) if u1 < u2 else (u2, u1)
				if tup in user_user:
					user_user[tup] += 1
				else:
					user_user[tup] = 1
	write_graph_to_file("user_user_subgraph.txt", user_user)

def get_pin_subgraph(pin_board_list):
	pin_pin = {}
	for p1 in pin_board_list:
		for b in pin_board_list[p1]:
			for p2 in board_p_list[b]:
				tup = (p1, p2) if p1 < p2 else (p2, p1)
				if (p1, p2) in pin_pin:
					pin_pin[tup] += 1
				else:
					pin_pin[tup] = 1

	write_graph_to_file("pin_pin_subgraph.txt", pin_pin)

#this one gives three type of board-board graphs
#1 using connections based on common users
#2 using connections based on common pins
#3 using all common connections over users and pins
def get_board_subgraphs(board_u_list, board_p_list):
	board_board_u = {}
	board_board_p = {}
	for b1 in board_u_list:
		for u in board_u_list[b1]:
			for b2 in user_board_list[u]:
				tup = (b1, b2) if b1 < b2 else (b2, b1)
				if tup in board_board_u:
					board_board_u[tup] += 1
				else:
					board_board_u[tup] = 1
	write_graph_to_file("uboard_uboard_subgraph.txt", board_board_u)

	for b1 in board_p_list:
		for u in board_p_list[b1]:
			for b2 in user_board_list[u]:
				tup = (b1, b2) if b1 < b2 else (b2, b1)
				if tup in board_board_u:
					board_board_u[tup] += 1
				else:
					board_board_u[tup] = 1
				if tup in board_board_p:
					board_board_p[tup] += 1
				else:
					board_board_p[tup] = 1
	write_graph_to_file("pboard_pboard_subgraph.txt", board_board_p)
	write_graph_to_file("board_board_subgraph.txt", board_board_u)

def write_graph_to_file(filename, edgeset):
	print "writing file" + filename
	f=open(filename, "w")
	for e in edgeset:
		line = str(e[0]) + "\t" + str(e[1]) + "\t" + str(edgeset[e]) + "\n"
		f.write(line)
	f.close()

main()
