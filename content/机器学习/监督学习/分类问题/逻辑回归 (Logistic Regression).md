---
title: 逻辑回归
draft: "false"
tags:
  - 机器学习
---
## 问题

给定**目标变量**y（离散取值，如0，1）在**特征**x的一组取值$(x^{(1)},\dots,x^{(m)})$下的值$(y^{(1)}, \dots, y^{(m)})$，求在新的特征变量取值下y属于各个类别的概率

## 思路

### 假设函数

使用**sigmoid函数**（也称logistic函数）将线性组合映射到[0,1]区间：
$$
h_\theta(x) = g(\theta^T x) = \frac{1}{1+e^{-\theta^T x}}
$$
其中：
$$
\theta^T x = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n
$$

**解释**：
- $h_\theta(x)$表示$P(y=1|x;\theta)$，即给定x和参数θ下y=1的概率
- 当$\theta^T x > 0$时，$h_\theta(x) > 0.5$，预测y=1
- 当$\theta^T x < 0$时，$h_\theta(x) < 0.5$，预测y=0
- **决策边界**：$\theta^T x = 0$的超平面将特征空间分为两个区域

### 代价函数

不能使用均方误差，因为会导致非凸函数。使用**对数似然损失**：
$$
Cost(h_\theta(x), y) = \begin{cases}
-\log(h_\theta(x)) & \text{if } y = 1 \\
-\log(1-h_\theta(x)) & \text{if } y = 0
\end{cases}
$$

简化形式：
$$
Cost(h_\theta(x), y) = -y\log(h_\theta(x)) - (1-y)\log(1-h_\theta(x))
$$

总代价函数：
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)})) \right]
$$

### 最小化代价函数

#### 梯度下降法

梯度计算（注意：形式与线性回归相同但h_θ不同）：
$$
\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

矩阵形式：
$$
\nabla J(\theta) = \frac{1}{m} X^T (h_\theta(X) - y)
$$

更新规则：
$$
\theta := \theta - \alpha \nabla J(\theta)
$$

#### 高级优化算法

- **共轭梯度法 (Conjugate Gradient)**：比梯度下降收敛更快
- **BFGS**：拟牛顿法，自动选择学习率
- **L-BFGS**：有限内存BFGS，适用于大规模问题

## 多类分类

### 一对多 (One-vs-All)

对K类问题，训练K个二元分类器：
- 分类器i：将类别i与其他所有类别区分
- 预测时选择输出概率最高的分类器对应的类别

## 优化

### 特征缩放

与线性回归相同，使用标准化或归一化

### 正则化

**L2正则化（岭回归）**：
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2
$$

**L1正则化（Lasso）**：
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)})) \right] + \frac{\lambda}{m}\sum_{j=1}^{n}|\theta_j|
$$

## 推广

逻辑回归中，输入变量直接作为特征，使得模型的能力有限，可以考虑变换输入变量后再作为特征输入，从而提高模型的能力：
- 引入中间层变换输入变量：[[神经网络 (Neural Networks)]]
- 加入核函数变换输入变量：[[支持向量机 (SVM)]]