import matplotlib.pyplot as plt

import pickle

file = "data/avgDefUtil_msg_1000.pkl"
# file = "data/avgDefUtil_rand_1000.pkl"

with open(file) as f:
	[avgDefUtilList, numEpisode] = pickle.load(f)


plt.plot(avgDefUtilList)
plt.show()




























