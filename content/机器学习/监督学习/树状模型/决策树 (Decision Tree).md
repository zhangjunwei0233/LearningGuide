---
title: 决策树 (Decision Tree)
draft: "false"
tags:
  - 机器学习
---

## 一、适用场景

生活中有很多数据具有**离散的输入属性**，比如视频数据集的题材，导演，演员等属性。在这些输入上的机器学习输出过程可以被建模为一个问答序列，所有可能的序列构成一棵**决策树**，节点为属性，边为对应属性的离散取值。

![[Pasted image 20260110152020.png|500]]

**决策树具有可解释性强、易于构建的优点**。对于一个这样的数据集，如何构建一棵准确、高效的决策树便成为了一个机器学习问题。

## 二、构建算法

### 1. 基础算法

对于一个离散输入属性的数据集，构建一棵决策树的普遍思路如下：
1. 选择一个属性作为根节点，依据可能的取值建立分支
2. 按照节点属性的取值，将数据集划分为多个子集，分支下每一个节点对应一个子集
3. 对每个叶节点，递归选择属性、划分子集
4. 重复如上步骤直到一个节点的子集里，全部样本均属于同一类；或者属性用尽

伪代码描述如下：

```plaintext
procedure TREEGENERATE(D, A)
	Initialize: D = {(x1,y1), ..., (xn, yn)}, A = {a1, ..., ad}
	Generate node
	
	if all samples from D belong to class C then
		mark node as leaf node of class C; return
	end if
	if A = Phi or samples from D have some values on A then
		mark node as leaf node, class = most common class in D; return
	end if
	
	Chose the best split attribute a* from A
	for each values a*v in a* do
		create a branch for node; Dv denotes subset consists of samples from D have a*v
		use TREEGENERATE(Dv, A\{a*}) as branch node
	end for
end procedure
```

### 2. 选择最佳属性

#### a. 使用信息增益率

在构建分支的每一步，需要选择一个属性来拆分数据集。好的选择应该以简单的方式将数据集分开，使得后续分支的数据更加纯净。可以从熵的视角来量化这一标准。

考虑取值空间为$\mathcal{X}$的离散型随机变量$X$，**熵**可以衡量它取值的不确定性，其定义为：
$$
H(X) = - \sum_{x\in\mathcal{X}}P(X=x) \log_{2} P(X = x)
$$

若已知一个变量的部分信息，则它取值的不确定型会有所降低，此时熵变化为**条件熵**：
$$
\begin{align*}
H(Y|X) &= \sum_{x \in \mathcal{X}} P(X=x) H(Y|X=x) \\
&= - \sum_{x \in \mathcal{X}} P(X=x) \sum_{y\in \mathcal{Y}} P(Y=y|X=x) \log P(Y=y|X=x) \\
&= -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(X=x, Y=y) \log P(Y=y|X=x)
\end{align*}
$$

迁移到机器学习的语境下，考虑具有标签$\{ 1, \dots, K \}$和属性$\mathcal{A} = \{ A_{1}, \dots, A_{d} \}$的数据集$\mathcal{D}$，属性$A$取值$a$的子集记为$\mathcal{D}_{|A=a}$。则可以通过熵来衡量一个属性$A$对于该数据集的**信息增益 (Information Gain, IG)**：
$$
IG(\mathcal{D}, A) = H(\mathcal{D}) - H(\mathcal{D}|A)
$$
其中
$$
H( \mathcal{D}) = -\sum_{k = 1}^{K} p_{k} \log_{2}p_{k}
$$
$$
H(\mathcal{D}|A) = \sum_{a \in value(A)} \frac{|\mathcal{D}_{|A = a}|}{|\mathcal{D}|} H(\mathcal{D}_{|A = a})
$$
一个直观的想法为，在拆分的时候，最佳的属性应该选择最容易把数据集拆分的属性，也即信息增益最大的属性。

但是直接使用信息增益有一个问题，即**信息增益偏好于选择取值可能性多的属性**。可以将信息增益对该属性的**固有值 (intrinsic value, IV)** 做归一化得到**信息增益率 (Gain Ratio, GR)** 并以此为依据判断。

$$
GR(\mathcal{D}, A) = \frac{IG(\mathcal{D}, A)}{IV(\mathcal{D}, A)}
$$
$$
IV(\mathcal{D}, A) := - \sum_{a \in value(A)} \frac{|\mathcal{D}_{|A = a}|}{|\mathcal{D}|} \log_{2} \frac{|\mathcal{D}_{|A=a}|}{|\mathcal{D}|}
$$

#### b. 使用基尼系数

基尼系数的直观含义为从$\mathcal{D}$中随机取两个样本，两者属于不同类别的概率。值越大，不确定性越大。

整个数据集的基尼系数为：
$$
Gini(\mathcal{D}) = \sum_{k} p_{k} (1 - p_{k}) = 1 - \sum_{k} p_{k}^{2}
$$
某一属性$A$的基尼系数为：
$$
Gini(\mathcal{D}, A) = \sum_{a \in value(A)} \frac{|\mathcal{D}_{|A = a}|}{|\mathcal{D}|} Gini(\mathcal{D}_{|A=a})
$$
应该选择基尼系数小的属性。

### 3. 选择停止条件

在基础算法中，决策树构建的停止条件为所有属性用完或子类不可再细分。但是这样存在过拟合的风险，在实际应用中通常使用**剪枝策略**来防止过拟合

- **前剪枝**：当信息变得不可靠时，停止扩展分支
	- 使用验证集辅助，若拆分后验证机准确率降低，则不拆分
	- 可能导致欠拟合
- **后剪枝**：先生成完整的树，然后用叶子替换中间节点简化树结构
	- 使用验证集辅助，遍历所有内部节点，尝试替换为叶节点。若验证集准确率上升，则保留替换

一般而言后剪枝欠拟合风险小，泛化性更好，但是需要更多训练时间。

### 4. 应对连续属性与缺失属性

#### a. 连续属性离散化

对于连续属性，基本思路是将其离散化。

设连续属性$A$的一系列观测值为$a^{1},\dots,a^{n}$，则取候选拆分阈值为每相邻两观测值之间的中位数：
$$
T_{A} = \left\{   \frac{a^{i} + a^{i+1}}{2} \Big | 1 \leq i \leq n-1  \right\}
$$
选择信息增益最大的阈值作为二分阈值：

$$
\begin{align*}
IG(\mathcal{D}, A) &:= \max_{t \in T_{A}} IG(\mathcal{D}, A, t) \\
& = \max_{t \in T_{A}} \left\{  H(\mathcal{D}) - \frac{|\mathcal{D}_{|A \geq t}|}{|\mathcal{D}|}H(\mathcal{D}_{|A\geq t}) -  \frac{|\mathcal{D}_{|A \geq t}|}{|\mathcal{D}|}H(\mathcal{D}_{|A< t}) \right\}
\end{align*}
$$

#### b. 将缺失样本按属性比例平均到已知样本

给定数据集$\mathcal{D}$和属性$A$，令$\tilde{\mathcal{D}}$为数据集中该属性值没有缺失的样本子集。给每一个样本设定权重$w_{x}$，并定义
- 总体缺失比例：$\rho = \frac{\sum_{x \in \tilde{\mathcal{D}}}w_{x}}{\sum_{x \in \mathcal{D}}w_{x}}$ 为属性未缺失的样本权重比例
- 类别$k$的比例：$\tilde{p}_{k}= \frac{\sum_{x \in \tilde{\mathcal{D}}_{|y = k}}w_{x}}{\sum_{x \in \tilde{\mathcal{D}}}w_{x}}$ 为属性未缺失的样本中，第$k$类所占的权重比例
- 属性取值$a$的比例：$\tilde{r}_{a} = \frac{\sum_{x \in \tilde{\mathcal{D}}_{|A=a}}w_{x}}{\sum_{x \in \tilde{\mathcal{D}}}w_{x}}$ 为属性缺失的样本中，$A=a$的样本权重比例

则有缺失值的时候，属性$A$的信息增益为$IG(\mathcal{D}, A) = \rho \times IG(\tilde{\mathcal{D}}, A)$。若样本$x$的属性$a$没有缺失，则直接划分到对应子节点并保留权重$w_{x}$；否则把$x$同时划分到$a$下面的每一个子节点中，并把权重调整为$\tilde{r}_{a} \cdot w_{x}$ （即按照为缺失样本的属性值分布加权）。

## 三、随机森林

标准的决策树一次只能生成一棵树，随机森林将这个过程重复$m$次，每次抽取$n$个样本和$p$个属性构建决策子树，之后通过[[Bagging算法|Bagging算法]]集成这m棵树，以提高泛化能力。