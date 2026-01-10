---
title: Boosting算法
tags:
  - 机器学习
---
## 一、强可学习与弱可学习

- **强可学习**：一个概念是强可学习的若存在一个多项式学习算法可以学习它，且正确率很高
- **若可学习**：一个概念是弱可学习的若存在一个多项式学习算法可以学习它，但正确率仅比随即猜测好

可以证明，**强可学习和弱可学习是等价的**：

设对于分类任务，真实的分类器为$f$，基分类器为$h$,错误率为$\epsilon$，即：

$$
P(h_{i}(x) \neq f(x)) = \epsilon
$$

现在使用T个基分类器采用“多数投票”制构造集成分类器$H$，则$H$在$x$上的错误率为：
$$
\begin{align*}
P(H(x) \neq f(x)) &= \sum_{k = 0}^{\lfloor T/2 \rfloor }
\begin{pmatrix}
T \\
k
\end{pmatrix}
(1-\epsilon)^{k}\epsilon^{T-k} \\
& \leq \exp\left( -\frac{1}{2} T(1-2\epsilon)^{2} \right)
\end{align*}
$$
即集成分类器的错误率随着基分类器的错误率$\epsilon$和个数$T$递减。理论上只要$T$足够大，错误率可以控制地任意小。

基于一系列弱学习器构造强学习器的方法称为**集成学习 (Ensemble Learning)**，基本思路如下：
- 训练多个分类器，在训练地过程中动态改变训练样本地权重，让后面的分类器重点关注前面的分类器分类失败的样本。
- 即在每一轮，基于当前继承分类器的错误率重新赋予训练样本权重$D_{t}$。基于该权重最小化经验风险，得到一个新的弱分类器$h_{t}$。将这个若分类器的权重设置为$\alpha_{t}$，加入已有的集成分类器
- 最终的集成分类器为$H(x)=sign\left( \sum\alpha_{t}h_{t}(x) \right)$

## 二、AdaBoost

一个经典的集成学习算法

### 1. 任务定义

- 输入：
	- n个样本$\{ (x^{1}, y^{1}), \dots, (x^{n}, y^{n}) \}$，标签为1/-1标签
	- 弱学习器$h_{t}(x)$
- 最终的分类器：$H(x)  = \alpha_{1}h_{1}(x) + \dots + \alpha_{T}h_{T}(x)$
- 损失函数为指数损失函数：
$$
L_{exp}(H) = \sum_{i = 1}^{n} \exp(-y^{i}H(x^{i}))
$$

### 2. 问题求解

![[Pasted image 20260110190834.png|700]]

集成学习算法要解决的核心问题有损失函数的选择，每一轮$\alpha_{t}$的计算、数据分布$D_{t}$的更新和每一个弱学习器的优化目标。

#### a. 损失函数选择

上面已经指出AdaBoost使用指数损失函数，下面证明它的合理性，即优化指数损失函数可以达到贝叶斯最优分类：

$$
\frac{\partial L_{exp}(H)}{\partial H(x)} \Big |_{H^{*}} = -e^{-H^{*}(x)}P(Y=1|x) + e^{H^{*}(x)} P(Y=-1|x)
$$
令偏导等于0，解得：
$$
H^{*}(x) = \frac{1}{2} \ln \frac{P(Y=1|x)}{P(Y=-1|x)}
$$
故最终的预测为：
$$
\hat{y} =  sign(H^{*}(x)) = \begin{cases}
1,  & P(Y=1|x) > P(Y = -1|x) \\
-1,  & P(Y = 1|x) < P(Y = -1|x)
\end{cases}
$$
为贝叶斯最优分类器

#### b. $\alpha_{t}$的计算

$\alpha_{t}$的目标为最小化$L_{exp}(\alpha_{t}h_{t}|D_{t})$：
$$
\begin{align*}
L_{exp}(\alpha_{t}h_{t}|D_{t}) &= \mathbb{E}_{x \sim D_{t}}[e^{-y\alpha_{t}h_{t}(x)}] \\
& = e^{-\alpha_{t}}P_{x \sim D_{t}}(y = h_{t}(x)) + e^{\alpha_{t}}P_{x\sim D_{t}}(y \neq h_{t}(x)) \\
& = e^{-\alpha_{t}} (1 - \epsilon) + e^{\alpha_{t}} \epsilon_{t}
\end{align*}
$$
令其等于0，得到:
$$
\alpha_{t} = \frac{1}{2} \ln\left( \frac{1 - \epsilon_{t}}{\epsilon_{t}} \right)
$$
可见当$\epsilon_{t}$越大，$\alpha_{t}$就应该越小。当$\epsilon_{t} = 0.5$d的时候，$\alpha_{t}$减小到0。

#### c. 数据集更新与弱学习器优化

$h_{t+1}$应最小化$L_{exp}(H_{t} + \alpha_{t+1}h|D)$：

$$
\begin{align*}
L_{exp}(H_{t} + \alpha_{t + 1}h|D) &= \mathbb{E}_{x\sim D} [e^{-y(H_{t}(x) + \alpha_{t+1}h(x))}] \\
& \approx \mathbb{E}_{x \sim D} \left[ e^{-yH_{t}(x)}\left( 1 - y\alpha_{t+1}h(x)  \right) \right]
\end{align*}
$$
故
$$
\begin{align*}
h_{t+1} &= \arg \min_{h} L_{exp}(H_{t} + \alpha_{t+1}h|D) \\
&= \arg \max_{h} \mathbb{E}_{x \sim D} [e^{-yH_{t}(x)}yh(x)]
\end{align*}
$$
令
$$
D_{t+1}(x) \propto D_{t}(x) e^{-\alpha_{t}yh_{t}(x)}
$$
则可以使得$h_{t+1}$的优化目标转化为简单的0-1损失：
$$
h_{t+1} = \arg \min_{h} \mathbb{E}_{x \sim D_{t+1}}l_{0-1}(x, y)
$$

### 3. 优劣分析

AdaBoost算法的优势：
1. 速度快，需要优化的参数少
2. 灵活：可以结合任何学习算法
3. 可以**抵抗过拟合**：间隔理论的解释为，即使训练误差到达零，随着继续训练弱分类器，训练数据的间隔仍可以增大（关键在于动态调整了训练数据）

AdaBoost算法的局限性：
1. 需要选择合适的弱分类器：分类器过弱会导致欠拟合，过于复杂会导致过拟合
2. **容易受到噪声影响**
