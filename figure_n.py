from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
import matplotlib.pyplot as plt
import numpy as np 
plt.figure(figsize=(4.5,4))
base_accuracy = np.array([0.9602 for x in range(23)])
base_robustness = np.array([0.5910 for x in range(23)])
x1 = np.arange(0, 23, 1)
x2 = np.arange(0, 23, 1)
accuracy = np.array([0.9602,0.9598,0.9591,0.9580,0.959,0.9601,0.9579,0.9580,0.9577,0.9572,0.9597,0.9596,0.9567,0.9589,
                        0.9588,0.9576,0.9562,0.9553,0.9544,0.9529,0.9513,0.9501,0.8958])
robustness= np.array([0.5910,0.5081,0.4563,0.4449,0.4415,0.4361,0.4314,0.4286,0.4169,0.4079,0.3962,0.3953,0.3807
                        ,0.3427,0.3311,0.3304,0.3255,0.3126,0.2824,0.2516,0.2371,0.2370,0.1915])

host = host_subplot(111, axes_class=axisartist.Axes)
# plt.subplots_adjust(right=4 ,wspace=0.5, hspace=0.5)
par1 = host.twinx()
par1.axis["right"].toggle(all=True)


p1, = host.plot(x1, accuracy, color="red", zorder=15,label="Accuracy")
p1, = host.plot(x1, base_accuracy,color="red", zorder=15, label="Initial Accuracy",ls='--')
p2, = par1.plot(x2, robustness, color="blue", zorder=15,label="Robustness")
p2, = par1.plot(x2, base_robustness, color="blue", zorder=15,label="Initial Robutness",ls='--')
par1.fill_between(x1, robustness-0.07, robustness+0.07,facecolor='lightgray', alpha=0.1)
host.fill_between(x2, accuracy-0.02, accuracy+0.02,facecolor='lightgray', alpha=0.1)

host.set_xlim(0, 23)
host.set_ylim(0, 1)
par1.set_ylim(0, 1)


host.set_xlabel("Number of Bit Flip",fontsize=14,fontweight="bold")
host.set_ylabel("Accuracy",fontsize=14,fontweight="bold")
par1.set_ylabel("Robustness",fontsize=14,fontweight="bold")

host.legend()

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
#######################################
# base_accuracy = np.array([0.9629 for x in range(11)])
# base_robustness = np.array([0.5888 for x in range(11)])
# x1 = np.arange(0, 11, 1)
# x2 = np.arange(0, 11, 1)
# accuracy = np.array([0.9629,0.9616,0.9609,0.9464,0.9324,0.9289,0.9202,0.9223,0.9134,0.8231,0.8206])
# robustness = np.array([0.5888,0.243,0.2387,0.1943,0.1935,0.1884,0.1698,0.1654,0.1611,0.1442,0.1434])
# host = host_subplot(111, axes_class=axisartist.Axes)
# par1 = host.twinx()
# par1.axis["right"].toggle(all=True)


# p1, = host.plot(x1, accuracy, color="red", zorder=15,label="Accuracy")
# p1, = host.plot(x1, base_accuracy,color="red", zorder=15, label="BaseAccuracy",ls='--')
# p2, = par1.plot(x2, robustness, color="blue", zorder=15,label="Robustness")
# p2, = par1.plot(x2, base_robustness, color="blue", zorder=15,label="BaseRobustness",ls='--')
# par1.fill_between(x1, robustness-0.09, robustness+0.09,facecolor='lightgray', alpha=0.1)
# host.fill_between(x2, accuracy-0.02, accuracy+0.02,facecolor='lightgray', alpha=0.1)

# host.set_xlim(0, 11)
# host.set_ylim(0, 1)
# par1.set_ylim(0, 1)


# host.set_xlabel("Number of Bit Flip",fontsize=20,fontweight="bold")
# host.set_ylabel("Accuracy",fontsize=20,fontweight="bold")
# par1.set_ylabel("Robustness",fontsize=20,fontweight="bold")

# host.legend(loc=(0.4,0.30))

# host.axis["left"].label.set_color(p1.get_color())
# par1.axis["right"].label.set_color(p2.get_color())    
plt.savefig("lenet_end.eps")                    
plt.show()