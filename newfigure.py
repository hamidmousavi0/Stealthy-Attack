import numpy as np
import matplotlib.pyplot as plt
# base_accuracy = np.array([0.9629 for x in range(11)])
# base_robustness = np.array([0.5888 for x in range(11)])
base_accuracy = np.array([0.9602 for x in range(23)])
base_robustness = np.array([0.5910 for x in range(23)])
x1 = np.arange(0, 23, 1)
x2 = np.arange(0, 23, 1)
accuracy = np.array([0.9602,0.9598,0.9591,0.9580,0.959,0.9601,0.9579,0.9580,0.9577,0.9572,0.9597,0.9596,0.9567,0.9589,
                        0.9588,0.9576,0.9562,0.9553,0.9544,0.9529,0.9513,0.9501,0.8958])
robustness= np.array([0.5910,0.5081,0.4563,0.4449,0.4415,0.4361,0.4314,0.4286,0.4169,0.4079,0.3962,0.3953,0.3807
                        ,0.3427,0.3311,0.3304,0.3255,0.3126,0.2824,0.2516,0.2371,0.2370,0.1915])
# accuracy = np.array([0.9629,0.9616,0.9609,0.9464,0.9324,0.9289,0.9202,0.9223,0.9134,0.8231,0.8206])
# robustness = np.array([0.5888,0.243,0.2387,0.1943,0.1935,0.1884,0.1698,0.1654,0.1611,0.1442,0.1434])
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x1, accuracy, color="red", zorder=15, label="Accuracy")
ax2.plot(x2, robustness, color="blue", zorder=15, label="Robustness")
ax1.plot(x1, base_accuracy, color="red", zorder=15, label="Base Accuracy",ls='--')
ax2.plot(x2, base_robustness, color="blue", zorder=15, label="Base Robustness ",ls='--')
ax2.fill_between(x1, robustness-0.09, robustness+0.09,facecolor='blue', alpha=0.1)
ax1.fill_between(x2, accuracy-0.04, accuracy+0.04,facecolor='red', alpha=0.1)
ax2.set_xlabel('number of bit flips', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax2.set_ylabel('Robustness', fontsize=12)
ax1.set_xlim(xmin=0, xmax=23)
ax2.set_xlim(xmin=0, xmax=23)
ax1.set_ylim(ymin=0, ymax=1)
ax2.set_ylim(ymin=0, ymax=1)
ax1.legend()
ax2.legend()
ax1.grid(True)
ax2.grid(True)
# ax1.grid(which='major', alpha=0.5, linestyle='--')
# ax2.grid(which='major', alpha=0.5, linestyle='--')
plt.show()