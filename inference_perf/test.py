import matplotlib.pyplot as plt

# 1. 填入您的两组压测数据
qps = [5, 10, 15, 20, 25, 30, 35]

# 第一组：k8s-service (RR) 数据（秒换算为毫秒）
k8s_service = [777, 1155, 1875, 4137, 12306, 32268, 50178]

# 第二组：gateway-baseline 数据（秒换算为毫秒）
gateway_baseline = [201, 205, 207, 213, 239, 394, 619]

# 2. 开始绘图
plt.figure(figsize=(9, 6), dpi=100)

# 绘制两条曲线，配色和样式参考了您上传的图片
plt.plot(qps, k8s_service, label='k8s-service (RR)',
         color='#ff7f0e', marker='o', markersize=8, linewidth=2)
plt.plot(qps, gateway_baseline, label='gateway-baseline',
         color='#1f77b4', marker='o', markersize=8, linewidth=2)

# 3. 各种图表美化元素（完全对齐参考图风格）
plt.title('TTFT p90 vs QPS', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('QPS (requested rate)', fontsize=12, labelpad=10)
plt.ylabel('TTFT p90 (ms)', fontsize=12, labelpad=10)

# 添加灰色虚线网格
plt.grid(True, linestyle=':', alpha=0.6, color='gray')

# 调整刻度字体
plt.xticks(qps, fontsize=11)
plt.yticks(fontsize=11)

# 4. 显示图例
plt.legend(loc='upper left', frameon=True, fontsize=11, facecolor='white', edgecolor='lightgray')

# 5. 自动调整布局并弹出窗口
plt.tight_layout()
plt.show()