---
title: 支持向量机 (Support Vector Machine, SVM)
draft: "false"
tags:
  - 机器学习
---
## 一、最大化间隔以提高泛化性

对于逻辑回归问题，可以将分类边界$\{ x|w^{T}x + b =0\}$定义为**分类超平面**，在平面一侧的样本被分类为0，另一侧为1。放缩$(w,b)$得到的不同模型对应同样的分类超平面，为了解决这个多对一的问题，定义使得距离分类超平面最近的样本满足$|w^{T}x + b| = 1$的$(w,b)$的超平面为**典型分类超平面**。

定义距离超平面最近的输入向量为**支持向量**，支持向量到超平面的距离为**间隔 (Margin)**。理论上，**更大的间隔意味着更好的泛化性**，因此分类任务的目标不光是学习得到正确的分类边界，还需要最大化该分类边界的间隔。

间隔$m$的数学计算如下：

![[Pasted image 20260110111204.png|300]]

$$
\begin{align*}
m &= \frac{1}{2} \times \left( \frac{w}{\| w \|} \right)^{T} (x^{(+)} - x^{(-)}) \\
&= \frac{1}{2 \|w\|} \times (w^{T}x^{(+)} - w^{T}x^{(-)}) \\
&= \frac{1}{2 \|w\|} \times (1 - (-1)) \\
&= \frac{1}{\|w\|}
\end{align*}
$$
因此，最大化间隔等价于最小化$\|w\|$，其中$(w,b)$是典型分类超平面。结合分类任务本来的要求，新的任务可以形式化定义为，称为**SVM优化问题**：

$$
\begin{align*}
& \min_{w} \|w\|^{2} \\
\text{s.t.} \quad & y^{(i)}(w^{T}x^{(i)} + b) \geq 1, i = 1, \dots, n
\end{align*}
$$

## 二、求解方法

### 1. 拉格朗日乘值法

对于问题
$$
\begin{align*}
& \min_{w} f(w) \\
\text{s.t.} \quad & g_{i}(w) \leq 0, i = 1, \dots, k \\
& h_{j}(w) = 0, j = 1, \dots, l
\end{align*}
$$

构造其拉格朗日型为：

$$
L_{p}(w, \alpha, \beta) = f(w) + \sum_{i = 1}^{k}\alpha_{i}g_{i}(w) + \sum_{j = 1}^{l} \beta_{j}h_{j}(w)
$$

则原始问题等价于：

$$
\min_{w} \max_{\alpha\geq 0, \beta} L_{p}(w, \alpha, \beta)
$$

### 2. 强对偶定理

称$\min_{w} \max_{\alpha\geq 0, \beta} L_{p}(w, \alpha, \beta)$的对偶问题为：
$$
\max_{\alpha\geq 0, \beta} \min_{w} L_{p}(w, \alpha, \beta)
$$
两问题不同，但是若原问题满足强对偶定理条件：

1. $f(w)$和$g_{i}(w)$对于$w$是凸函数
2. $h_{i}(w)$是仿射函数，即$h(w) = a^{T}w + b$
3. $\exists w_{0}, \forall i, g_{i}(w_{0}) < 0$

则两问题等价，并且一组参数是原始问题和对偶问题的解当且仅当该参数满足**Karush-Kuhn-Tucker, KKT条件**：

- $\partial L_{p}(w, \alpha, \beta) = 0$，平稳性条件
- $\forall i, \alpha_{i}g_{i}(w) = 0$，互补松弛条件
- $\forall i, g_{i}(w) \leq 0$，原始不等式约束
- $\forall i, h_{i}(w) = 0$，原始等式约束
- $\forall i, \alpha_{i} \geq 0$，对偶问题中约束条件

### 3. 求解SVM优化问题

首先使用拉格朗日乘值法转化原始问题为：
$$
\begin{align*}
& \min_{w, b} \max_{\alpha} L_{p} (w, b, \alpha) \\
& L_{p}(w, b, \alpha) = \frac{1}{2} \|w\|^{2} + \sum_{i = 1}^{n} \alpha_{i} [1 - y^{(i)}(w^{T}x^{(i)} + b)]
\end{align*}
$$
又由于该问题满足强对偶定理条件，故可进一步转化为其对偶问题：
$$
\max_{\alpha} \min_{w,b} L_{p}(w, b, \alpha) := \max_{\alpha} L_{d}(\alpha)
$$
其中$L_{p}(w, b, \alpha)$称为**Hinge Loss**，为SVM问题的损失函数，$L_{d}(\alpha)$为最小化的损失函数，可以通过梯度下降法求解，但在这里我们讨论他的解析解：
$$
L_{d}(\alpha) = \min_{w,b} L_{p}(w,b,\alpha)
$$
令$L_{p}(w, b, \alpha)$对于$w,b$的偏导为零，可得：
$$
\begin{cases}
w = \sum_{i = 1}^{n} \alpha_{i}y^{(i)} x^{(i)} \\
\sum_{i = 1}^{n} \alpha_{i} y^{(i)} = 0
\end{cases}
$$
代回可得：
$$
\begin{align*}
L_{d}(\alpha) &= -\frac{1}{2} w^{T}w + \sum_{i = 1}^{n}\alpha_{i} \\
&= -\frac{1}{2}\sum_{i}\sum_{j}\alpha_{i} \alpha_{j} y^{(i)}y^{(j)}(x^{(i)})^{T}x^{(j)} + \sum_{i = 1}^{n}\alpha_{i}
\end{align*}
$$
接下来只需求解：
$$
\begin{align*}
& \max_{\alpha \in \mathbb{R}^{n}, \alpha \geq 0} \left( -\frac{1}{2} \sum_{i}\sum_{j} \alpha_{i} \alpha_{j} y^{(i)} y^{(j)} (x^{(i)})^{T} x^{(j)} + \sum_{i = 1}^{n}\alpha_{i} \right) \\
\text{s.t.} \quad & \sum_{i = 1}^{n}\alpha_{i}y^{(i)} = 0
\end{align*}
$$
这仍然是一个二次规划问题，但是负责度取决于样本数$n$而不再是样本维度$d$，可是使用**SMO算法**高效求解。

之后只需要反解出$w$和$b$即可。$w$已经由下式给出

$$
w = \sum_{i = 1}^{n} \alpha_{i}y^{(i)}x^{(i)}
$$
$b$则可以通过KKT条件的互补松弛条件确定：
$$
\alpha_{i} [1 - y^{(i)} (w^{T}x^{(i)} + b)] = 0
$$
若$\alpha_{i} > 0$，则$1-y^{(i)}(w^{T}x^{(i)} + b) = 0$，样本位于典型分离超平面上，是支持向量；若$\alpha_{i} = 0$，则样本非支持向量。

故选择出全部的支持向量，然后计算出b即可。分类超平面由支持向量决定，这就是算法**支持向量机 (Support Vector Machine, SVM)** 的由来。

> [!note] SVM loss与逻辑回归loss的对比
> 
> $$
> l_{SVM} = \sum_{i = 1}^{n}[1 - y^{(i)}(w^{T}x^{(i)} + b)]_{+} + \lambda \|w\|^{2}
> $$
> $$
> l_{LR} = -\sum_{i = 1}^{n} \log \sigma(y^{(i)}(w^{T}x^{(i)} + b)) + \lambda\|w\|^{2}
> $$
> 
> 可以看出二者完全可类比，故SVM也属于经验风险最小化 (ERM) 问题。

## 三、优化

### 1. 处理非线性数据——Kernel SVM

若数据非线性可分，则无法找到一个超平面作为合理的决策边界。此时核心的想法是进行**数据升维**：通过非线性变换$\phi$将数据映射到高维空间，使得其在高维空间中变得线性可分，从而化归回SVM。

![[Pasted image 20260110121851.png|300]]

如此一来，原始的优化问题变为：
$$
\begin{align*}
& \min_{w} \|w\|^{2} \\
\text{s.t.} \quad & y^{(i)}(w^{T}\phi (x^{(i)}) + b) \geq 1, i = 1, \dots, n
\end{align*}
$$

对应的对偶优化问题和判别函数变为：
$$
\begin{align*}
& \max_{\alpha \in \mathbb{R}^{n}, \alpha \geq 0} \left( -\frac{1}{2} \sum_{i}\sum_{j} \alpha_{i} \alpha_{j} y^{(i)} y^{(j)} \phi(x^{(i)})^{T} \phi (x^{(j)}) + \sum_{i = 1}^{n}\alpha_{i} \right) \\
\text{s.t.} \quad & \sum_{i = 1}^{n}\alpha_{i}y^{(i)} = 0
\end{align*}
$$
$$
f(x) = \sum_{x^{(i) \in \mathcal{SV}}} \alpha_{i} y^{(i)} \phi(x^{(i)})^{T} \phi(x) + b
$$

注意到在实际求解的时候用到的都是$\phi(x^{(i)})^{T}\phi(x)$这样的内积形式，故无需具体定义$\phi$，而只需要定义其内积即可。该内积函数称为**核函数**$K(\cdot, \cdot)$，它将两个向量映射为对称连续的实数值：
$$
K(x, x') = K(x', x), \forall x, x' \in \mathbb{R}^{d}
$$
常见的核函数有：

![[Pasted image 20260110123524.png|500]]

直观上，核函数衡量了两个输入向量之间的相似度。**Mercer定理**给出了判定函数是否为核函数的充分条件。

> [!note]- Mercer定理
> 假设函数$K: [a, b] \times [a,b] \to \mathbb{R}$是对称半正定核函数，即对于任意样本集$\{ x^{(1)}, \dots, x^{(n)} \}$，其对应的核矩阵$K$是对称半正定矩阵，那么存在正交基函数$\{ e_{j} \in L^{2}[a,b]; j = 1, \dots \}$，使得：
> $$
> K(x, x') = \sum_{j = 1}^{\infty} \lambda_{j} e_{j}(x) e_{j}(x')
> $$

### 2. 处理数据异常——软间隔

错误标注的数据会严重影响SVM找到的分类超平面的泛化性。解决的思路为添加**松弛变量**$\xi$，允许对困难或噪声样本错误分类。这样的间隔称为**软间隔**。

原始的优化任务变为：
$$
\begin{align*}
& \min_{w} \frac{1}{ 2}\|w\|^{2} + C\sum_{i = 1}^{N} \xi_{i} \\
\text{s.t.} \quad & y^{(i)}(w^{T}x^{(i)} + b) \geq 1 - \xi_{i}, \forall i, \xi_{i} \geq 0
\end{align*}
$$
$\xi_{i}$的含义为：
- 若$\xi_{i} = 0$，则$x^{(i)}$在分离超平面之外，被正确分类
- 若$0 < \xi_{i} < 1$，则$x^{(i)}$虽然被正确分类，但是在分离超平面内，没有把握
- 若$\xi_{i} > 1$，则$x^{(i)}$被错误分类

引入的$C$为正则化超参数，用于调整软间隔的程度。$C$越大，越接近硬间隔SVM，支持向量越少。