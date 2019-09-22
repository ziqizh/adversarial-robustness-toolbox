import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.switch_backend('agg')

log1 = open('data-log/measure/atta-m-loss-default.log')
log2 = open('data-log/measure/atta-mnist-mat-atta-1.log')

label1 = "TRADES ATTA-1"
label2 = "MAT ATTA-1"

data1 = []
data2 = []
length = 20

log_lines1 = log1.readlines()
log_lines2 = log2.readlines()

for i in range(length):
  data1.append([eval(j) for j in log_lines1[i].split(' ')])
  data2.append([eval(j) for j in log_lines2[i].split(' ')])

print(len(data1))

x = np.array([i[0] for i in data1]) + 1

adv_loss1 = np.array([i[1] for i in data1])
adv_loss2 = np.array([i[1] for i in data2])


current_palette = sns.color_palette()

plt.plot(x, adv_loss1, color=current_palette[0], label=label1, lw=2)
plt.plot(x, adv_loss2, color=current_palette[1], label=label2, lw=2)
# plt.plot(x, adv_loss3, color=current_palette[2], label=label3, lw=2)
# plt.plot(x, adv_loss4, color=current_palette[3], label=label4, lw=2)

plt.xlabel("Attack iteration(k,m)", fontsize=15)
plt.ylabel("Loss after converged", fontsize=15)
plt.tick_params(labelsize=10)

plt.legend(fontsize='x-large')

plt.savefig('data-pic/robust-accuracy-by-epoch.png')