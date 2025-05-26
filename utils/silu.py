import torch
import numpy as np
import matplotlib.pyplot as plt

# 创建输入数据
x = torch.linspace(-6, 6, 10000)

# 计算SiLU: x * sigmoid(x)
silu = x * torch.sigmoid(x)

# 创建图表
plt.plot(x.numpy(), silu.numpy(), 'b-', label='SiLU', linewidth=2)
# 设置图表属性
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('SiLU(x)', fontsize=12)
plt.title('SiLU Activation Function', fontsize=14)
plt.legend(fontsize=12)

# 设置坐标轴范围
plt.xlim(-6, 6)
plt.ylim(-2, 6)

print(silu)
plt.show()
