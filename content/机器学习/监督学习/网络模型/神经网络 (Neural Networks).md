---
title: 神经网络
tags:
  - 机器学习
draft: "false"
---
神经网络是由简单的（通常是自适应的）元素，及其分层组织组成的大规模、并行、互连网络，旨在以类似于生物神经系统的方式，与现实世界的对象进行交互。

## 一、感知机

感知机 (Perceptron) 由线性函数和非线性激活函数组成：
$$
y = \sigma\left( \sum_{i = 1}^{n} x_{i}w_{i} - w_{0} \right)
$$
其中非线性激活函数$\sigma$为sigmoid函数，输入输出不限于布尔值，权重和阈值可以通过分析或者学习算法来确定。

感知机可以正确表示布尔逻辑AND/OR/NOT：
- 与逻辑AND：$x_{1} \wedge x_{2}: \{ (0, 0;0), (0, 1;0),  (1, 0;0), (1, 1;1) \}$
	- 令$w_{1}=w_{2}=1, w_{0}=1.5$，则$y = 1$当且仅当$x_{1}=x_{2}=1$
- 或逻辑OR：$x_{1} \vee x_{2}: \{ (0, 0;0), (0, 1;1),  (1, 0;1), (1, 1;1) \}$
	- 令$w_{1}=w_{2}=1, w_{0}=0.5$，则$y=1$当且仅当$x_{1}=1$或$x_{2}=1$
- 非逻辑NOT：$\neg x_{1}: \{ (0;1), (1;0) \}$
	- 令$w_{1}=-1, w_{2}=0,w_{0}=-0.5$，则$y$为$x_{1}$的非

![[Pasted image 20260111114225.png|500]]

一个很自然的问题就是：**感知机能表示异或 XOR 吗？**

异或问题是线性不可分的问题，因此单个感知机不能表示异或，需要添加隐藏层：

![[Pasted image 20260111114446.png|500]]

## 二、前馈神经网络

由前面的内容可知，想要提高感知机的表达能力，需要构建多层感知机结构，这就形成了前馈神经网络：

![[Pasted image 20260111114841.png|300]]

**前馈神经网络**也称为深度前馈网络或多层感知机，是典型的深度学习模型，目标是逼近特定函数$f^{*}(x)$ （通常是Bayes最优决策函数）
- **前馈**：信息从输入$x$流经用于定义函数$f$的中间计算过程最终到达输出$y$，不存在将模型输出反馈回自身的连接结构
- **网络**：由不同函数复合而成，函数复合的方式由有向无环图描述

### 1. 网络架构

#### a. 隐层单元

神经网络的每一层对输入做线性仿射变换后计算其非线性激活值。完整这一流程的单元称为隐层单元$h = g(W^{T}x + c)$。非线性的激活函数有很多选择，常见的有：
- **ReLU及其扩展**：$g(z) = \max\{ 0, z \} + \alpha \min\{ 0, z \}$
	- $\alpha=0$：标准ReLU
	- $\alpha = -1$：绝对值矫正
	- $\alpha = 0.01$：Leaky ReLU
	- $\alpha$待学习：PReLU
- **Sigmoid函数**：$g(z) = \sigma(z) = \frac{e^{z}}{1 + e^{z}}$
	- Sigmoid激活函数在两侧趋于平缓，可能导致深层的神经网络在训练中**梯度消失**
- **Tangent函数**：$g(z) = \tanh(z)$
	- $\tanh(z) = 2\sigma(2z) - 1$

#### b. 输出单元

在神经网络中，最后输出的隐藏层表征$h$需要经过额外的变换才能得到满足任务要求的输出。完成这一变换的单元称为**输出单元**。对应不同任务，常见的输出单元有：
- **线性单元 (回归任务)**：$\hat{y} = W^{T}h + b$
	- 假设输出$y$服从条件高斯分布$p(y|x) = \mathcal{N}(y;\hat{y},I)$，此时最大化对数似然等价于$L_{2}$损失函数，因此损失函数一般使用$L_{2}$损失
- **Sigmoid单元 (二分类任务)**：$\hat{y} =\sigma(w^{T}h + b)$
	- 假设输出$y$服从二项分布$B(n, \hat{y})$。此时最大化对数似然等价于最小化Logistic损失
- **Softmax单元 (多分类任务)**：$\hat{y} = softmax(W^{T}h + b)$
	- 假设输出$y$服从多项分布$M(n, \hat{y})$，最大化对数似然等价于最小化交叉熵损失


#### c. 宽度与深度

- **宽度**：根据**通用近似定理**，带有sigmoid类型激活函数的单隐层和线性输出单元的前馈神经网络，随着隐藏层宽度增加，可以任意精度逼近Borel可测函数。
- **深度**：相比于浅层神经网络，深层神经网络实现相同表达力所需要的单元数显著减少。
### 2. 训练算法

#### a. 损失函数

与其他参数模型相同，模型的损失函数选择为**负对数条件似然/交叉熵损失**：
$$
R_{n}(\theta) = - \mathbb{E}_{x\sim \hat{p}_{x, data}} \mathbb{E}_{y \sim \hat{p}_{y|x, data}} \log p_{\theta}(y|x)
$$
在特定的输出分布下，负对数似然损失会退化回其他特定的损失函数，如：
- 若数据分布满足高斯分布$p_{\theta}(\cdot|x) = \mathcal{N}(f(x;\theta),I)$，交叉熵损失函数退化为$L_{2}$损失函数，且其贝叶斯最优决策为条件期望$\mathbb{E}(y|x)$
$$
R_{n}(\theta) = \frac{1}{2} \mathbb{E}_{x,y \sim \hat{p}_{data}} \| y - f(x;\theta) \|_{2}^{2} + const
$$
- 若数据分布满足拉普拉斯分布，则此时损失函数退化为$L_{1}$损失函数，贝叶斯最优决策变为条件中位数
$$
R_{n}(\theta) = \mathbb{E}_{x, y \sim \hat{p}_{data}} \| y - f(x;\theta) \|_{1}
$$
- 若数据分布满足二项分布$B(f(x;\theta))$，损失会退化到Logistic损失，贝叶斯最优决策变为最大后验概率
$$
R_{n}(\theta) = -[y\log f(x;\theta) + (1-y)\log(1 - f(x;\theta))]
$$
- 若数据分布满足多项分布$M(1, f(x;\theta))$，损失会退化为交叉熵损失，贝叶斯最优决策为最大后验概率
$$
R_{n}(\theta) = - \sum_{c = 1}^{C} y_{c}\log f(x;\theta)
$$

> [!note] 对于损失函数的理解
> 先前遇到过的绝大多数损失函数（除了SVM使用的Hinge Loss）全都可以看作是**负对数似然损失**在特定概率分布假设下的退化形式。

#### b. 反向传播 (Back Propagation, BP) 算法

反向传播算法是训练神经网络的核心算法，它基于**链式法则**（chain rule）高效计算损失函数相对于所有参数的梯度。算法的核心思想可以概括为"**前向传播计算输出，反向传播计算梯度**"。

在神经网络中，损失函数 $L$ 是网络输出的复合函数：
1. **前向传播**：从输入层到输出层，逐层计算激活值
	$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = g(z^{(l)})$$
	其中 $a^{(0)} = x$（输入），$a^{(L)} = \hat{y}$（输出）

2. **反向传播**：从输出层到输入层，逐层计算梯度
   - 输出层误差：
	$$\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}} = \frac{\partial L}{\partial a^{(L)}} \odot g'(z^{(L)})$$
   - 隐藏层误差：
	$$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}=\frac{\partial L}{\partial z^{(l+1)}} \frac{\partial z^{(l+1)}}{\partial a^{(l)}} \frac{\partial a^{(l)}}{\partial z^{(l)}}=(W^{(l+1)T} \delta^{(l+1)}) \odot g'(z^{(l)})$$
   - 参数梯度：
	$$
     \frac{\partial L}{\partial W^{(l)}} =\frac{\partial L}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial W^{(l)}}= \delta^{(l)} a^{(l-1)T}, \quad \frac{\partial L}{\partial b^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial b^{(l)}}=\delta^{(l)}$$

算法流程的流程可以表示为：

```text
for each training example (x, y):
    1. 前向传播：计算网络输出 ŷ
    2. 计算损失 L(ŷ, y)
    3. 反向传播：计算梯度 ∂L/∂θ
    4. 更新参数：θ ← θ - η·∂L/∂θ
```

#### c. 防止过拟合——正则化

神经网络中常见的正则化方法有：
- **添加惩罚参数**：$\tilde{J}(\theta;X,y) = J(\theta; X,y) + \lambda \Omega(\theta)$
	- $L_{2}$正则化：$\Omega(\theta) = \frac{1}{2} \| w \|_{2}^{2}$，等价于权重衰减
	- $L_{1}$正则化：$\Omega(\theta) = \| w \|_{1}$
	- 带约束的优化：$\min_{\theta} J(\theta; X, y) \quad \text{s.t.} \quad \Omega(\theta) \leq k$
- **早停**：记录并更新验证集损失最小的模型
- **Dropout**，**参数共享**，**稀疏表示**，**模型集成**等等

#### d. 优化算法

常用的优化方法有：
- **随机梯度下降 (SGD)**：
$$
\begin{align*}
\hat{g} & \leftarrow \frac{1}{|S|} \nabla_{\theta} \left( \sum_{s\in S} L(f(x^{(s)};\theta), y^{(s)}) \right) \\
\theta & \leftarrow \theta - \eta_{t} \hat{g}
\end{align*}
$$
- **带动量的随机梯度下降 (SGDM)**：将历史梯度数据指数衰减累计
$$
\begin{align*}
\hat{g} & \leftarrow \frac{1}{|S|} \nabla_{\theta} \left( \sum_{s\in S} L(f(x^{(s)};\theta), y^{(s)}) \right) \\
v & \leftarrow \alpha v - \eta_{t} \hat{g} \\
\theta & \leftarrow \theta + v
\end{align*}
$$
- **自适应学习率方法**：学习率随时间、维度变化
	- **AdaGrad**:
		$$
		\begin{align*}
		r & \leftarrow r + g \odot g \\
		\theta & \leftarrow \theta - \frac{\eta_{t}}{\delta + \sqrt{ r }} \odot g
		\end{align*}
		$$
	- **RMSProp**：在AdaGrad上引入滑动平均
		$$
		\begin{align*}
		r & \leftarrow \rho r + (1 - \rho) g \odot g \\
		\theta & \leftarrow \theta - \frac{\eta_{t}}{\sqrt{ \delta + r }} \odot g
		\end{align*}
		$$
	- **Adam**：在RMSProp上引入动量
		$$
		\begin{align*}
		s & \leftarrow \rho_{1} s + (1 - \rho_{1})g, & \hat{s} = \frac{s}{1 - \rho_{1}^{t}}\\
		r & \leftarrow \rho_{2} r + (1 - \rho_{2}) g \odot g, & \hat{r} = \frac{r}{1 - \rho_{2}^{t}} \\
		\theta & \leftarrow \theta - \frac{\eta_{t}}{\delta + \sqrt{ \hat{r} }} \odot \hat{s}
		\end{align*}
		$$

## 三、改进架构

[[卷积神经网络 (CNN)]]

[[循环神经网络 (RNN)]]
