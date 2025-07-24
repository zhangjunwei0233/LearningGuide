---
title: 神经网络
tags:
  - 机器学习
draft: "false"
---
## 问题

当特征数量很大时，传统算法计算复杂度过高；当需要学习复杂非线性关系时，多项式特征会导致特征爆炸

## 基本思路

### 神经元模型

神经网络由多层神经元组成：
- **输入层 (Input Layer)**：接收原始特征
- **隐藏层 (Hidden Layer)**：进行特征变换
- **输出层 (Output Layer)**：产生最终预测

### 前向传播 (Forward Propagation)

设第$l$层有$s_l$个神经元，激活函数为$g(z)$：

$$
a^{(l+1)} = g(\Theta^{(l)} a^{(l)})
$$

其中：
- $a^{(l)}$：第$l$层的激活值
- $\Theta^{(l)}$：第$l$层到第$l+1$层的权重矩阵
- $g(z)$：激活函数（通常为sigmoid函数）

**详细计算过程**：
$$
z^{(l+1)} = \Theta^{(l)} a^{(l)} \quad (\text{加上bias项})
$$
$$
a^{(l+1)} = g(z^{(l+1)})
$$

### 代价函数

对于K类分类问题：
$$
J(\Theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} \left[ y_k^{(i)} \log(h_\Theta(x^{(i)}))_k + (1-y_k^{(i)}) \log(1-(h_\Theta(x^{(i)}))_k) \right]
$$
$$
+ \frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} (\Theta_{ji}^{(l)})^2
$$

### 最小化代价函数——反向传播算法 (Backpropagation)

1. **前向传播**：计算所有层的激活值
2. **计算输出层误差**：$\delta^{(L)} = a^{(L)} - y$
3. **反向传播误差**：
   $$
   \delta^{(l)} = (\Theta^{(l)})^T \delta^{(l+1)} \cdot g'(z^{(l)})
   $$
4. **计算梯度**：
   $$
   \frac{\partial J}{\partial \Theta_{ij}^{(l)}} = \frac{1}{m} a_j^{(l)} \delta_i^{(l+1)} + \lambda \Theta_{ij}^{(l)} \quad (j \neq 0)
   $$
   $$
   \frac{\partial J}{\partial \Theta_{i0}^{(l)}} = \frac{1}{m} a_0^{(l)} \delta_i^{(l+1)} \quad (\text{bias项})
   $$
5. **之后可使用梯度下降法处理**

## 优化

### 权重初始化

**随机初始化**：避免对称性破缺
$$
\Theta_{ij}^{(l)} \sim \text{Uniform}[-\epsilon, \epsilon]
$$
其中 $\epsilon = \sqrt{\frac{6}{s_l + s_{l+1}}}$

### 梯度检验 (Gradient Checking)

使用数值方法验证反向传播的正确性：
$$
\frac{\partial J}{\partial \Theta} \approx \frac{J(\Theta + \epsilon) - J(\Theta - \epsilon)}{2\epsilon}
$$

### 网络架构选择

- **隐藏层数**：通常1-3层足够
- **隐藏单元数**：通常比输入特征数多，但要防止过拟合
- **输出层**：
  - 二分类：1个单元
  - 多分类：K个单元（K个类别）
