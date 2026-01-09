---
title: 逻辑回归
draft: "false"
tags:
  - 机器学习
---
## 一、问题

给定**目标变量**y（离散取值，如0，1）在**特征**x的一组取值$(x^{(1)},\dots,x^{(m)})$下的值$(y^{(1)}, \dots, y^{(m)})$，求在新的特征变量取值下y属于各个类别的概率

## 二、思路

### 1. 假设函数

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

### 2. 代价函数

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

### 3. 最小化代价函数

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

## 三、多类分类

### 1. 化归为二分类

一种想法是将多分类问题化归为二分类问题，有两种思路：

- **一对多 (One-vs-All, OVA)**：预测是否为第k类
	1. 选择一个二分类算法
	2. $\forall k \in \{ 0, \dots, K-1 \}$，将第$k$类样本设为正例，其余样本设为负例，训练二分类打分函数$f_{k}$
	3. 对于输入$x$，预测其类别为$\arg \max_{k} f_{k}(x)$
	4. 这种方法的问题为训练时正例与负例数量不平衡
- **一对一 (One-vs-One, OVO)**：预测是第k类还是第k'类
	1. 选择一个二分类算法
	2. $\forall k, k' \in \{ 0, \dots, K-1 \}$，将第$k$类样本设为正例，第$k'$类样本设为负例，训练二分类打分函数$f_{k, k'}$
	3. 对于输入$x$，预测其类别为$\arg \max_{k} \left( \sum_{k' \neq k}f_{k, k'}(x) \right)$
	4. 这种方法的问题为分类器数量大

![[Pasted image 20260109230742.png|500]]

### 2. Softmax多分类器

另一种想法是将多分类问题看作一个独立的问题，训练一个**softmax多分类器**直接解决该问题。设对于输入$x$，模型的输出$Wx+b$不再是单独的一个值，而是一个长度为类别数量的向量。将这个向量通过softmax函数即可得到每个类别的概率：
$$
p_{w}(x) = softmax(Wx + b)
$$

为了优化softmax多分类器，需要以**交叉熵 (CE)** 为损失。对于两个概率分布$p=\{ p_{1}, \dots, p_{k} \}$和$q=\{ q_{1}, \dots, q_{K} \}$，交叉熵定义为：
$$
H(p,q) = \sum_{k=1}^{K}p_{k}\log q_{k}
$$
交叉熵可以反应$q$分布相对于$p$分布的偏差值。在多分类任务中，$p$为训练集独热标签，$q$为模型预测的概率分布，故交叉熵损失函数定义为：

$$
l_{CE}(x,y;W) = \sum_{k = 1}^{K} \mathbb{I}_{[y=k]}\log \frac{\exp((W^{(k)})^{T}x)}{\sum_{k'=1}^{K} \exp((W^{(k')})^{T}x)}
$$

### 3. 应对类别不平衡问题

如果各类别的数据比例差距过大，则会出现精度悖论 (BaseRate高，但是没有预测能力)。一般有两种处理思路：

- **代价敏感预测**：改变划分标签的阈值
	- 将预测$y=1$的阈值从$\frac{t}{1-t} > 1$改为$\frac{t}{1-t} > \frac{n^{+}}{n^{-}}$
- **采样方法**：平衡各类别的训练数据
	- **欠采样法**：随机去掉大类样本。如EasyEnsemble
	- **过采样法**：扩增小样本。如SMOTE

## 四、优化

### 1. 特征缩放

与线性回归相同，使用标准化或归一化

### 2. 正则化

- **L2正则化（岭回归）**：
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2
$$

- **L1正则化（Lasso）**：
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)})) \right] + \frac{\lambda}{m}\sum_{j=1}^{n}|\theta_j|
$$

## 五、推广

逻辑回归中，输入变量直接作为特征，使得模型的能力有限，可以考虑变换输入变量后再作为特征输入，从而提高模型的能力：

- 引入中间层变换输入变量：[[神经网络 (Neural Networks)]]
- 加入核函数变换输入变量：[[支持向量机 (SVM)]]