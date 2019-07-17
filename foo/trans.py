from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np


import collections
import itertools
import time

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
import matplotlib.pyplot as plt

from scipy import spatial
import tensorflow as tf

from collections import deque

def getDefaultGraph0():

	g0 = nx.DiGraph()
	g0.add_node(0, isDef=0, isAtt=0, isUAV=0, r=1, d=1, ctr=0)
	g0.add_node(1, isDef=0, isAtt=0, isUAV=0, r=0, d=0, ctr=0)
	g0.add_node(2, isDef=0, isAtt=0, isUAV=0, r=2, d=2, ctr=0)
	g0.add_node(3, isDef=0, isAtt=0, isUAV=0, r=1, d=2, ctr=0)

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


g = getDefaultGraph0()

def _create_feature(attr, fields):
	   return np.hstack([np.array(attr[field], dtype=float) for field in fields])

def _gtmp2intmp(gtemp):
	intmp = gtemp.copy()
	inNodeFields = ["isDef","isAtt","isUAV","r","d"]

	for nodeIndex, nodeFeature in gtemp.nodes(data=True):
		intmp.add_node(nodeIndex, 
					   features=_create_feature(nodeFeature, inNodeFields))
		
	for nodeIndex, nodeFeature in intmp.nodes(data=True):
		for attr in nodeFeature.keys():
			if attr != "features":
				del nodeFeature[attr]


	for u, v, features in gtemp.edges(data=True):
		intmp.add_edge(u, v, features=[0.0]) #TODO: change this to []

	return intmp


def _gtmp2ttmp(gtemp):
	ttmp = gtemp.copy()
	
	for nodeIndex, nodeFeature in gtemp.nodes(data=True):
		ttmp.add_node(nodeIndex, features=[])
		
	for nodeIndex, nodeFeature in ttmp.nodes(data=True):
		for attr in nodeFeature.keys():
			if attr != "features":
				del nodeFeature[attr]


	for u, v, features in gtemp.edges(data=True):
		ttmp.add_edge(u, v, features=[0.0]) #TODO: change this to []

	return ttmp


































