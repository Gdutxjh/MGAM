
import matplotlib.font_manager as fm
# 示例数据
import matplotlib.pyplot as plt
import matplotlib as mpl



# 示例数据
x = [36,54,72,96]
# y1 = [76.7,76.86,76.92,76.84,76.98,77.1,77.15,77.14,77.16,77.34,77.57,77.63,77.60,77.58,77.59,77.57,77.57,77.56,77.53,77.55,77.57,77.53,77.56,77.55,77.52,77.56,77.51,77.53,77.55,77.57]
y2 = [72.51, 68.74, 74.28, 66.30]
fig, ax = plt.subplots(figsize=(8, 8))
# 创建折线图
# plt.plot(x, y1, label = 'Twitter-2015')
ax.plot(x, y2, label = 'Twitter-2017')

# 设置自定义刻度和标签
custom_ticks = [36,54,72,96]

ax.set_xticks(custom_ticks)

# 添加标题和轴标签
ax.set_title("Twitter-2017", fontsize = 25)
ax.set_xlabel("置信度阈值", fontsize = 28)
ax.set_ylabel("预测准确率(ACC%)", fontsize = 28)

# 显示图形
plt.show()
fig.savefig(f'figure.png', dpi=600)

