---
title: 词向量 (Word2Vect)
draft: "false"
tags:
  - NLP
---
## 基本思路

使用单词的**上下文**（相邻单词）构建**语义**（词向量）

记全文$V$中共有$T$个单词，第$i$个位置的单词记为$w_{i}$，其语义由k个参数$\theta$组成的向量$u_{w}$表达：

- 作为center word时，单词构成的矩阵为$V_{nr\_words \times k}$
- 作为outside word时，单词构成的矩阵为$U_{nr\_words \times k}$

使用点乘计算两个单词的**相近程度**，$p(w_{t+j}|w_{t})$表示在$t$位置为$w_{t}$的前提下，$t+j$位置为$w_{t+j}$的概率，可以由相近程度计算 (设中心词为c, 外部词为o)：
$$
P(o|c) = softmax(u_{o}^{T}v_{c}) =  \frac{\exp(u_{o}^{T}v_{c})}{\sum_{w \in V} \exp(u_{w}^{T}v_{c})}
$$

定义单词**上下文窗口**的大小为$m$，则全文所有单词的“自洽程度”定义为各个单词在其上下文中出现的概率的积：
$$
Likelihood=L(\theta)= \prod_{t=1}^{T} \prod_{-m \leq j\leq m} P(w_{t+j}|w_{t};\theta)
$$
训练的目标就是为每个单词找到合理的词向量$\theta$，使得总自洽程度$L(\theta)$最大化

## 代价函数

$$
J(\theta) = - \frac{1}{T} \log L(\theta)= - \frac{1}{T} \sum_{t=1}^{T}\sum_{-m \leq j\leq m} \log P(w_{t+j}|w_{t}; \theta)
$$
最大化$L(\theta)$，只需要最小化$J(\theta)$

> [!note] 代价函数的偏导数
> $$
> \frac{\partial J}{\partial v_{c}} = U (\hat{y} - y)
> $$
> $$
> \frac{\partial J}{\partial U} = v_{c} (\hat{y} - y)^{T}
> $$

> [!NOTE]- 推导：
> 
> $$
> \begin{array}{l}
> J(\theta) &= - \log (P(O=o|C=c)) = -\log \left(  \frac{\exp{u_{o}^{T}v_{c}}}{\sum \exp{u_{w}^{T} v_{c}}} \right) \\
> &= - u^{T}_{o} v_{c} + \log \left( \sum \exp u^{T}_{w} v_{c} \right) 
> \end{array}
> $$
> $$
> \begin{array}{l}
> \frac{\partial J}{\partial v_{c}} &= - u_{o} + \frac{1}{\sum \exp(u^{T}_{w}v_{c})} \cdot \sum \exp (u_{w}^{T}v_{c}) u_{w} \\
> &= -u_{o} + \sum \hat{y}_{w} u_{w} \\
> &= - Uy + U\hat{y} \\
> &= U (\hat{y} - y)
> \end{array}
> $$
> 当$w = o$时：
> $$
> \frac{\partial J}{\partial u_{w}} =  -v_{c} + \frac{1}{\sum \exp (u^{T}_{w} v_{c})}  \cdot \exp (u_{w}^{T} v_{c}) v_{c} = -v_{c} + \hat{y}_{w}v_{c} = (\hat{y}_{w} - 1)v_{c}
> $$
> 当$w \neq o$时：
> $$
> \frac{\partial J}{\partial u_{w}} = \frac{1}{\sum \exp (u^{T}_{w} v_{c})}  \cdot \exp (u_{w}^{T} v_{c}) v_{c} = \hat{y}_{w}v_{c}
> $$
> 
> 综合两种情况：
> $$
> \frac{\partial J}{\partial U} = v_{c} (\hat{y} - y)^{T}
> $$
> 

## 最小化代价函数——随机梯度下降 (Stochastic Gradient Descent, SGD)

## 优化

### 使用negative sampling替换softmax代价函数

计算softmax代价函数时需要对所有的$\exp(u^{T}_{w}v_{c})$求和，计算量过大。
采用随机抽样的方法优化，任意选出一些不相关的词 (negative samples)计算损失。损失函数更新如下：

$$
J_{neg-sample}(u_{o}, v_{c}, U) = - \log(sigmoid(u^{T}_{o}v_{c})) -  \sum_{k \in \{ K\ sampled\ indices \}}\log (sigmoid(- u^{T}_{k}v_{c}))
$$