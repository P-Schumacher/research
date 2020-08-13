import numpy as np
from pudb import set_trace
from matplotlib import pyplot as plt

def search_last(trace):
    for x in trace[::-1]:
        if x != 0.:
            return x

plt.style.use('seaborn')
N = 150
fig, ax = plt.subplots()
x_s = []
y_s = []
color_iterable = []
for i in range(N):
    for j in range(10):
        trace = np.load(f'./visitation/visitation_{i}_{j}_TD3_AntMaze_hiro_ant_vis_plot_c10.npy')
        x = trace[:, 0]
        y = trace[:, 1]
        if i == 149:
            set_trace()
        x = search_last(x)
        y = search_last(y)
        x_s.append(x)
        y_s.append(y)
        color_iterable.append(i/float(N-1))

plot = ax.scatter(x_s, y_s, c=color_iterable, alpha=0.6, cmap='viridis')
fig.colorbar(plot)
plot = ax.scatter(0, 16, color='r')
ax.grid(False)
plt.xlim([-4, 20])
plt.ylim([-4, 20])
plt.tight_layout()
plt.xlabel('x - position')
plt.ylabel('y - position')
plt.show()


