import matplotlib.pyplot as plt

import pickle

file1 = "data/avgDefUtil_msg_1000.pkl"
file2 = "data/avgDefUtil_rand_1000.pkl"
file3 = "data/avgDefUtil_msgDdpg_1000.pkl"

with open(file1) as f:
	[l1, numEpisode] = pickle.load(f)


with open(file2) as f:
	[l2, numEpisode] = pickle.load(f)


with open(file3) as f:
	[l3, numEpisode] = pickle.load(f)


plt.plot(l1,"b")
plt.plot(l2,"r")
plt.plot(l3,"y")
plt.show()





# file = "data/avgDefUtilEps.pkl"

# with open(file) as f:
# 	[l, numEpisode] = pickle.load(f)

# plt.plot(l,"b")
# plt.show()

















