---
title: 多变量线性回归
draft: "false"
tags:
  - 机器学习
---

## 一、回归任务

在单变量线性回归的基础上，将特征变量个数扩展到n个$X_{1}, \dots, X_{n}$

## 二、思路

基本思路不变，假设函数推广为$h_\theta(x)=\theta_{0}+\theta_{1} x_{1} + \dots + \theta_{n}x_{n}$

矩阵形式：
$$
h_\theta(x) = \theta^T x = \begin{bmatrix}\theta_0 & \theta_1 & \cdots & \theta_n\end{bmatrix} \begin{bmatrix}1 \\ x_1 \\ \vdots \\ x_n\end{bmatrix}
$$

### 1. 代价函数

$$
J(\theta) = \frac{1}{2m}(X\theta - y)^T(X\theta - y)
$$

其中：
$$
X = \begin{bmatrix}
1 & x_{1}^{(1)} & \cdots & x_{n}^{(1)} \\
1 & x_{1}^{(2)} & \cdots & x_{n}^{(2)} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{1}^{(m)} & \cdots & x_{n}^{(m)}
\end{bmatrix}, \quad
\theta = \begin{bmatrix}
\theta_{0} \\
\theta_{1} \\
\vdots \\
\theta_{n}
\end{bmatrix}, \quad
y = \begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(m)}
\end{bmatrix}
$$

### 2. 梯度下降法

$$
\theta := \theta - \frac{\alpha}{m}X^T(X\theta - y)
$$

### 3. 正规方程法

$$
\theta = (X^{T}X)^{-1}X^{T}y
$$

## 三、优化

### 1. 特征缩放 (Feature Scaling)

不同特征的数值范围可能差异很大，需要标准化：

- **标准化 (Standardization)**：
$$
x_j = \frac{x_j - \mu_j}{\sigma_j}
$$

- **归一化 (Normalization)**：
$$
x_j = \frac{x_j - \min(x_j)}{\max(x_j) - \min(x_j)}
$$

### 2. 学习率选择

- 太小：收敛慢
- 太大：可能不收敛或发散
- 常用值：0.3, 0.1, 0.03, 0.01, 0.003, 0.001

### 3. 梯度下降 vs 正规方程

| 梯度下降      | 正规方程         |
| --------- | ------------ |
| 需要选择α     | 无需选择α        |
| 需要迭代      | 一步求解         |
| 适用大数据     | n < 10000时适用 |
| 复杂度O(kn²) | 复杂度O(n³)     |
