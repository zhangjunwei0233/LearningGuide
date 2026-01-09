---
title: 单变量线性回归
draft: "false"
tags:
  - 机器学习
---

## 一、问题

给定**目标变量**y在**特征**x的一组取值$(x^{(1)},\dots,x^{(m)})$下的值$(y^{(1)}, \dots, y^{(m)})$，求变量y关于特征x的函数关系$y=h_\theta(x)$，其中$h_\theta$称为**假设函数**（默认为线性关系）

## 二、思路

### 1. 假设函数

假设函数为$h_\theta(x)=\theta_{0}+\theta_{1}x$，其中$\theta_0$为偏置项（bias），$\theta_1$为权重

### 2. 代价函数

使用**均方误差**（Mean Squared Error, MSE）作为代价函数来衡量假设函数的准确程度：
$$
J(\theta_{0}, \theta_{1})= \frac{1}{2m} \sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2
$$
矩阵形式为：
$$
J(\theta)=\frac{1}{2m}(X\theta - y)^T(X\theta - y)
$$
其中：
$$
X = \begin{bmatrix}
1 & x^{(1)} \\
1 & x^{(2)} \\
\vdots & \vdots \\
1 & x^{(m)}
\end{bmatrix}, \quad
\theta = \begin{bmatrix}
\theta_{0} \\
\theta_{1}
\end{bmatrix}, \quad
y = \begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(m)}
\end{bmatrix}
$$

### 3. 最小化代价函数

#### 批量梯度下降法 (Batch Gradient Descent)

随机初始化$\theta_{0}, \theta_{1}$，每次沿着梯度**相反**的方向（负梯度方向）行进一小步，直至达到步数限制或收敛：

$$
\theta := \theta - \alpha \nabla J(\theta)
$$

其中学习率$\alpha > 0$，梯度为：
$$
\frac{\partial J}{\partial \theta_0} = \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})
$$
$$
\frac{\partial J}{\partial \theta_1} = \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})x^{(i)}
$$

矩阵形式：
$$
\theta := \theta - \frac{\alpha}{m}X^T(X\theta - y)
$$

#### 正规方程法 (Normal Equation)

通过令梯度为零直接求解最优参数：
$$
\nabla J(\theta) = 0 \Rightarrow \theta = (X^{T}X)^{-1}X^{T}y
$$

**注意事项**：
- 当$(X^{T}X)$不可逆时，通常是因为特征线性相关或特征数量大于样本数量
- 解决方法：删除冗余特征、使用正则化、或使用伪逆矩阵

**梯度下降 vs 正规方程**：
- 梯度下降：适用于大数据集，需要选择学习率
- 正规方程：适用于小数据集（n < 10000），无需迭代但计算复杂度为O(n³)

## 三、优化

### 正则化 (Regularization)

当模型复杂度过高时可能出现过拟合。通过对参数（除$\theta_{0}$外）添加惩罚项来缓解：

- **岭回归 (Ridge Regression, L2正则化)**：
$$
J(\theta)= \frac{1}{2m}\left[ \sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^{2} + \lambda \sum_{j=1}^{n}\theta_j^{2} \right]
$$

- **Lasso回归 (L1正则化)**：
$$
J(\theta)= \frac{1}{2m}\left[ \sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^{2} + \lambda \sum_{j=1}^{n}|\theta_j| \right]
$$

其中$\lambda \geq 0$为正则化参数：
- $\lambda = 0$：无正则化
- $\lambda$太大：可能导致欠拟合
- $\lambda$适中：在偏差和方差之间取得平衡

## 四、推广

1. 多变量推广：[[多变量线性回归 (Linear Regression with multiple variables)]]
2. 高次推广：[[多项式回归 (Polynomial Regression)]]