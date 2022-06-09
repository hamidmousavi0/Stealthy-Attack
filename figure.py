import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 11, 1)
base_accuracy = np.array([0.9629 for x in range(11)])
base_robustness = np.array([0.5888 for x in range(11)])
# accuracy = np.array([0.9602,0.9598,0.9591,0.9580,0.959,0.9601,0.9579,0.9580,0.9577,0.9572,0.9597,0.9596,0.9567,0.9589,
#                         0.9588,0.9576,0.9562,0.9553,0.9544,0.9529,0.9513,0.9501,0.8958])
accuracy = np.array([0.9629,0.9616,0.9609,0.9464,0.9324,0.9289,0.9202,0.9223,0.9134,0.8231,0.8206])
print(accuracy.shape)
# robustness= np.array([0.5910,0.5081,0.4563,0.4449,0.4415,0.4361,0.4314,0.4286,0.4169,0.4079,0.3962,0.3953,0.3807
#                         ,0.3427,0.3311,0.3304,0.3255,0.3126,0.2824,0.2516,0.2371,0.2370,0.1915])
robustness = np.array([0.5888,0.243,0.2387,0.1943,0.1935,0.1884,0.1698,0.1654,0.1611,0.1442,0.1434])
# error_robustness = np.random.normal(0.09, 0.02, size=robustness.shape)    
# error_accuracy = np.random.normal(0.01, 0.002, size=accuracy.shape)    
# robustness+=error_robustness  
# accuracy+=error_accuracy                  
print(robustness.shape)                        
plt.plot(x, accuracy, color="red", zorder=15, label="Accuracy")
plt.plot(x, base_accuracy, color="red", zorder=15, label="Base Accuracy",ls='--')
plt.plot(x, robustness, color="blue", zorder=15, label="Robustness")
plt.plot(x, base_robustness, color="blue", zorder=15, label="Base Robustness ",ls='--')
plt.scatter([3,3],[0.1943,0.9464],label="Drop Accuracy ")
plt.fill_between(x, robustness-0.06, robustness+0.06,facecolor='blue', alpha=0.1)
plt.fill_between(x, accuracy-0.03, accuracy+0.03,facecolor='red', alpha=0.1)
plt.xlabel('number of bit flips', fontsize=12)
plt.ylabel('Accuracy and Robutsness', fontsize=12)
plt.xlim(xmin=0, xmax=10)
plt.ylim(ymin=0, ymax=1)
plt.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', ncol=4, fontsize=8)
plt.grid(True)
plt.grid(which='major', alpha=0.5, linestyle='--')
plt.show()
                     
