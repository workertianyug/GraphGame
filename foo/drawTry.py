import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


""" return a default graph for use """
""" [isDef, isAtt, numUav, reward, penetration time] """
def getDefaultGraph0():

	g0 = nx.DiGraph()
	g0.add_node(0, isDef=0, isAtt=0, numUav=0, r=1, d=1, ctr=0)
	g0.add_node(1, isDef=0, isAtt=0, numUav=0, r=0, d=0, ctr=0)
	g0.add_node(2, isDef=0, isAtt=0, numUav=0, r=2, d=2, ctr=0)
	g0.add_node(3, isDef=0, isAtt=0, numUav=0, r=1, d=2, ctr=0)

	g0.add_edge(0,0)
	g0.add_edge(1,1)
	g0.add_edge(2,2)
	g0.add_edge(3,3)
	
	g0.add_edge(0,1)
	g0.add_edge(1,0)
	g0.add_edge(1,2)
	g0.add_edge(2,1)
	g0.add_edge(2,3)
	g0.add_edge(3,2)
	g0.add_edge(0,3)
	g0.add_edge(3,0)
	g0.add_edge(0,2)
	g0.add_edge(2,0)

	g0.graph["utilDefC"] = 1
	g0.graph["utilAttC"] = -1

	return g0

fig, ax = plt.subplots(figsize=(6,4))
# ax.set_xticks([])
# ax.set_yticks([])

g = getDefaultGraph0()
pos = nx.spring_layout(g)

value = [0.0 for i in g.nodes()]



g.nodes(data=True)[0]["isDef"] = 1
g.nodes(data=True)[2]["isAtt"] = 1

value = []
for i in g.nodes:
	if g.nodes(data=True)[i]["isDef"]==1:
		value.append("blue")
	elif g.nodes(data=True)[i]["isAtt"]==1:
		value.append("red")
	else:
		value.append("black")

nx.draw_networkx_nodes(g, pos, node_color = value)
nx.draw_networkx_edges(g, pos, edge_color='black')
nx.draw_networkx_labels(g, pos)
circle2 = plt.Circle((0.5, 0.5), 0.2, color='blue')

ax.add_artist(circle2)

plt.axis("scaled")
print pos
plt.show(block=True)
plt.pause(1)
plt.clf()

# g.nodes(data=True)[1]["isDef"] = 1
# g.nodes(data=True)[3]["isAtt"] = 1
# value = []
# for i in g.nodes:
# 	if g.nodes(data=True)[i]["isDef"]==1:
# 		value.append("blue")
# 	elif g.nodes(data=True)[i]["isAtt"]==1:
# 		value.append("red")
# 	else:
# 		value.append("black")

# ax.set_xticks([])
# ax.set_yticks([])
# nx.draw_networkx_nodes(g, pos, node_color = value)
# nx.draw_networkx_edges(g, pos, edge_color='black')
# plt.show(block=False)
# plt.pause(3)
# plt.close()



















