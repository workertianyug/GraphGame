import matplotlib.pyplot as plt

import pickle

file1 = "data/5000/avgDefUtil_msg_5000.pkl"
file2 = "data/5000/avgDefUtil_rand_5000.pkl"
file3 = "data/5000/avgDefUtil_msgDdpg_5000.pkl"
file4 = "data/5000/avgDefUtil_maddpg_5000.pkl"

with open(file1) as f:
	[l1, numEpisode] = pickle.load(f)


with open(file2) as f:
	[l2, numEpisode] = pickle.load(f)


with open(file3) as f:
	[l3, numEpisode] = pickle.load(f)


with open(file4) as f:
	[l4, numEpisode] = pickle.load(f)


plt.plot(l1,"b") # gcn
plt.plot(l2,"r") # rand
plt.plot(l3,"y") # msg ddpg
plt.plot(l4,"c") # maddpg

plt.gca().legend(('gcn','rand','gcn+ddpg','maddpg'))
plt.show()





# file = "data/avgDefUtilEps.pkl"

# with open(file) as f:
# 	[l, numEpisode] = pickle.load(f)

# plt.plot(l,"b")
# plt.show()

















