import matplotlib.pyplot as plt

import pickle


def runningAvg(l, num):
	res = []

	for i in range(0, len(l)):
		if i < num:
			res.append(l[i])
		else:
			curTotal = 0
			for i in range(i - num, i):
				curTotal += l[i]
			res.append(curTotal/num)

	return res 


file1 = "data/5000_single/defUtil_paramRandDef_5000.pkl"

with open(file1) as f:
	[l1, numEpisode] = pickle.load(f)


l1 = runningAvg(l1,30)

plt.plot(l1,"b")



plt.gca().legend(('gcn'))
plt.show()





# file = "data/avgDefUtilEps.pkl"

# with open(file) as f:
# 	[l, numEpisode] = pickle.load(f)

# plt.plot(l,"b")
# plt.show()

















