import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def getDefaultGraph5x5():
	g = nx.DiGraph()

	for i in range(0,25):
		if i == 12:
			g.add_node(i, isDef=0, isAtt=0, numUav=0, r=3.0, d=3, ctr=0)
		elif i in [6,7,8,11,13,16,17,18]:
			g.add_node(i, isDef=0, isAtt=0, numUav=0, r=1.0, d=2, ctr=0)
		else:
			g.add_node(i, isDef=0, isAtt=0, numUav=0, r=0.0, d=0, ctr=0)

		g.add_edge(i,i)

	for i in range(0,24):
		if i in [4,9,14,19,24]:
			continue
		g.add_edge(i,i+1)
		g.add_edge(i+1,i)

	for i in range(0,4):
		for j in range(0,5):
			g.add_edge(i*5 + j, i*5+5 + j)
			g.add_edge(i*5+5 + j, i*5 + j)			

	g.graph["utilDefC"] = 2.0
	g.graph["utilAttC"] = -2.0

	pos = dict()
	pos[0] = np.array([0.0,4.0])
	pos[1] = np.array([1.0,4.0])
	pos[2] = np.array([2.0,4.0])
	pos[3] = np.array([3.0,4.0])
	pos[4] = np.array([4.0,4.0])
	pos[5] = np.array([0.0,3.0])
	pos[6] = np.array([1.0,3.0])
	pos[7] = np.array([2.0,3.0])
	pos[8] = np.array([3.0,3.0])
	pos[9] = np.array([4.0,3.0])
	pos[10] = np.array([0.0,2.0])
	pos[11] = np.array([1.0,2.0])
	pos[12] = np.array([2.0,2.0])
	pos[13] = np.array([3.0,2.0])
	pos[14] = np.array([4.0,2.0])
	pos[15] = np.array([0.0,1.0])
	pos[16] = np.array([1.0,1.0])
	pos[17] = np.array([2.0,1.0])
	pos[18] = np.array([3.0,1.0])
	pos[19] = np.array([4.0,1.0])
	pos[20] = np.array([0.0,0.0])
	pos[21] = np.array([1.0,0.0])
	pos[22] = np.array([2.0,0.0])
	pos[23] = np.array([3.0,0.0])
	pos[24] = np.array([4.0,0.0])

	return g,pos


g,pos = getDefaultGraph5x5()


fig, ax = plt.subplots(figsize=(6,4))


g.nodes(data=True)[0]["isDef"] = 1
g.nodes(data=True)[2]["isAtt"] = 1

value = []
for i in g.nodes:
	if g.nodes(data=True)[i]["isDef"]==1:
		value.append("blue")
	elif g.nodes(data=True)[i]["isAtt"]==1:
		value.append("red")
	else:
		value.append("gray")


nx.draw_networkx_nodes(g, pos, node_color = value)
nx.draw_networkx_edges(g, pos, edge_color='black')
nx.draw_networkx_labels(g, pos)
plt.axis("scaled")

plt.show(block=True)
























