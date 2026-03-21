import numpy as np
import matplotlib.pyplot as plt

# 1. 定义目标函数（需要被拟合的真理）
def target_function(x):
    return np.sin(x)

# 2. 数据采集
# 在 [-pi, pi] 之间采样 1000 个点作为数据集
X = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
Y = target_function(X)

# 划分训练集和测试集 (80% 训练, 20% 测试)
indices = np.random.permutation(len(X))
train_idx, test_idx = indices[:800], indices[800:]
X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

# 3. 模型描述（纯 NumPy 实现两层网络）
class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        # 使用 He 初始化（适合 ReLU）
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2./input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2./hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, x):
        # 前向传播
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, x, y, output, lr):
        # 反向传播 (链式法则)
        m = x.shape[0]
        
        # 1. 计算输出层误差
        dz2 = output - y # MSE 导数
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 2. 计算隐藏层误差 (经过 ReLU)
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 3. 更新权重 (SGD)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

# 4. 训练模型
model = SimpleNeuralNet(input_size=1, hidden_size=128, output_size=1)
epochs = 40000
learning_rate = 0.01

losses = []
for epoch in range(epochs):
    # 前向传播
    pred = model.forward(X_train)
    # 计算损失 (MSE)
    loss = np.mean((pred - Y_train)**2)
    losses.append(loss)
    # 反向传播
    model.backward(X_train, Y_train, pred, learning_rate)
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# 5. 拟合效果可视化
Y_pred = model.forward(X_test)

plt.figure(figsize=(10, 5))
# 绘制原始函数
plt.scatter(X_test, Y_test, color='blue', label='Real Data (Test Set)', s=10)
# 绘制拟合结果
plt.scatter(X_test, Y_pred, color='red', label='Neural Net Prediction', s=10)
plt.title("Function Fitting: Sine Wave using 2-layer ReLU Network")
plt.legend()
plt.show()