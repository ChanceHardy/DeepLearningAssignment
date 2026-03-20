import numpy as np
import matplotlib.pyplot as plt

# 1. 生成数据
X = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
Y = np.sin(X) + np.random.normal(0, 0.05, X.shape) # 加入噪声

# 2. 初始化两层网络参数 (Xavier 初始化)
W1 = np.random.randn(1, 20) * np.sqrt(2/1)
b1 = np.zeros((1, 20))
W2 = np.random.randn(20, 1) * np.sqrt(2/20)
b2 = np.zeros((1, 1))

# 3. 训练循环 (手动求导)
lr = 0.01
for i in range(5000):
    # 前向传播
    h1 = np.dot(X, W1) + b1
    h1_relu = np.maximum(0, h1) # ReLU
    y_pred = np.dot(h1_relu, W2) + b2
    
    # 计算 Loss ( MSE )
    loss = np.mean((y_pred - Y)**2)
    
    # 反向传播
    grad_y_pred = 2.0 * (y_pred - Y) / X.shape[0]
    grad_W2 = np.dot(h1_relu.T, grad_y_pred)
    grad_b2 = np.sum(grad_y_pred, axis=0)
    
    grad_h1_relu = np.dot(grad_y_pred, W2.T)
    grad_h1 = grad_h1_relu.copy()
    grad_h1[h1 <= 0] = 0 # ReLU 梯度
    
    grad_W1 = np.dot(X.T, grad_h1)
    grad_b1 = np.sum(grad_h1, axis=0)
    
    # 更新参数
    W1 -= lr * grad_W1
    W2 -= lr * grad_W2
    b1 -= lr * grad_b1
    b2 -= lr * grad_b2

# 4. 可视化结果
plt.scatter(X, Y, label='Target')
plt.plot(X, y_pred, color='red', label='Prediction')
plt.legend()
plt.show()