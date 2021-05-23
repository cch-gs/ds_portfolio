import matplotlib.pyplot as plt

# plot
plt.plot([1,2,3,4], [2,4,6,8])
plt.plot([1,2,3,4], [1,2,3,4])
plt.xlabel('X-Label')
plt.ylabel('Y-Label')
plt.savefig('./graph/plot.png', dpi=300)
plt.show()
plt.close()

# color
plt.plot([1,2,3,4], [2,4,6,8], 'r--')
plt.plot([1,2,3,4], [1,2,3,4], 'm-.')
plt.xlabel('X-Label')
plt.ylabel('Y-Label')
plt.savefig('./graph/color&style.png', dpi=300)
plt.show()
plt.close()

# grid
plt.plot([1,2,3,4], [2,4,6,8])
plt.plot([1,2,3,4], [1,2,3,4])
plt.grid(True)

plt.savefig('./graph/grid.png', dpi=300)
plt.show()
plt.close()

# line
plt.plot([1,2,3,4], [2,4,6,8])
plt.plot([1,2,3,4], [1,2,3,4])
plt.axhline(y=3, color='r', linestyle='--',  linewidth=1)
plt.axvline(x=2, color='b', linestyle='-.',  linewidth=1)

plt.savefig('./graph/lines.png', dpi=300)
plt.show()
plt.close()

# bar graph
x = [0,1,2]
years = ['2020', '2021', '2022']
prices = [500, 450, 700]

plt.bar(x, prices)
plt.xticks(x, years)
plt.savefig('./graph/bar.png', dpi=300)
plt.show()
plt.close()

# barh graph
x = [0,1,2]
years = ['2020', '2021', '2022']
prices = [500, 450, 700]

plt.barh(x, prices, tick_label=years)
plt.savefig('./graph/barh.png', dpi=300)
plt.show()
plt.close()

# scatter
import numpy as np
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
size = (100 * np.random.rand(N))
plt.scatter(x, y, s=size, c=colors, alpha=0.5)
plt.show()
plt.savefig('./graph/scatter.png', dpi=300)
plt.close()

# histogram
# weight = [68, 81, 64, 56, 78, 74, 61, 77, 66, 68, 59, 71, 80, 59, 67, 81, 69, 73, 69, 74, 70, 65]
height = [170, 165, 174, 176, 173, 167, 185, 184, 183, 174, 191, 164, 153, 153, 175, 175, 178, 156, 146, 185, 175, 177]
plt.hist(height)
# plt.hist(weight, bins=100, density=True, alpha=0.7, histtype='step')
# plt.hist(height, bins=50, density=True, alpha=0.5, histtype='stepfilled')
plt.savefig('./graph/histogram.png', dpi=300)
plt.show()
plt.close()

# pie
ratio = [27, 33, 15, 25]
labels = ['Bear', 'Soju', 'Wine', 'Makuly']
plt.pie(ratio, labels=labels, autopct='%.2f%%')
plt.show()
plt.savefig('./graph/pie.png', dpi=300)
plt.close()