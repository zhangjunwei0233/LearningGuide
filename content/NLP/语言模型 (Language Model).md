---
title: RNN
draft: "false"
tags:
  - NLP
---
## 语言模型的概念

给定一列单词$(x^{(t)}, \dots, x^{(1)})$，语言模型的任务是预测下一个可能出现的单词的概率分布：
$$
P(x^{(t+1)}|x^{(t)}, \dots, x^{(1)}),\ \ x\in V= \{ w_{1}, \dots, w_{|V|} \}
$$
从另一个角度讲，语言模型可以给出一段文字的概率。因为：
$$
P(x^{(1)}, \dots, x^{(T)}) = P(x^{(1)}) \times P(x^{(2)} | x^{(1)}) \times \dots \times P(x^{(T)} | x^{(T-1)}, \dots, x^{(1)}) = \prod_{t=1}^{T} P(x^{(t)} | x^{(t-1)}, \dots, x^{(1)})
$$

## 语言模型的实现

### n-gram Language Model

n-gram Language Model 是一种最简单的语言模型。它只通过每个单词的前$n-1$个单词来预测该单词，即将单词预测简化为一个**马尔可夫过程**，进而通过**条件概率公式**使用频率预测概率：
$$
\begin{array}{l}
P(x^{(t+1)} | x^{(t)}, \dots, x^{(1)}) &= P(x^{(t+1)} | x^{(t)}, \dots, x^{t - n + 2}) \\
&= \frac{P(x^{(t+1)}, x^{(t)}, \dots, x^{t - n + 2})}{P(x^{(t)}, \dots, x^{t - n + 2})} \\
& \approx \frac{Count(x^{(t+1)}, x^{(t)}, \dots, x^{t - n + 2})}{Count(x^{(t)}, \dots, x^{t - n + 2})}
\end{array}
$$

### Recurrent Neural Networks (RNN)

n-gram Language Model 的马尔可夫假设严重地限制了模型的语义理解能力，因此我们需要构造一种可以输入文本任意长，关键词在不同位置出现都等价的语言模型。

![[Pasted image 20250728172133.png|700]]

#### 梯度消失与梯度爆炸

在上述的向前传播过程中，每一位置$t$的单词预测都会产生误差。设模型预测分布为$\hat{y}^{(t)}$，实际分布为$y^{(t)}$ (one-hot)，使用**交叉信息熵**衡量误差：
$$
J^{(t)}(\theta) = CE(\hat{y}^{(t)}, y^{(t)}) = - \sum_{w \in V} y^{(t)}_{w} \log \hat{y}^{(t)}_{w} = -\log \hat{y}^{(t)}_{x_{t+1}}
$$
模型在一个单词序列上的平均误差为：
$$
J(\theta) = \frac{1}{T} \sum_{t=1}^{T} J^{(t)}(\theta) = - \frac{1}{T} \sum_{t=1}^{T} \log \hat{y}^{(t)}_{x_{t+1}}
$$
考虑模型的反向传播，在计算$\partial J / \partial W_{h}$的时候，需要层层计算到最初的$W_{h}$层，再将每一层的梯度求和：
$$
\frac{\partial J^{(t)}}{\partial W_{h}} =  \sum_{i = 1}^{t} \frac{\partial J^{(t)}}{\partial W_{h}} \Bigg|_{(i)}
$$
这在传播链路过长（单词序列过长）的时候会出现问题，如 (为简化计算，假设不套用非线性的sigmoid函数)：
$$
\begin{array}{l}
\frac{\partial J^{(4)}}{\partial h(1)} &= \frac{\partial h_{2}}{\partial h_{1}} \cdot \frac{\partial h_{3}}{\partial h_{2}} \cdot \frac{\partial h_{4}}{\partial h_{3}} \cdot \frac{J^{(4)}}{\partial h_{4}} \\
&= \frac{\partial J^{(4)}}{\partial h_{4}} W_{h}^{3}
\end{array}
$$
可以观察到出现了$W_{h}$的高次方项。即若$W_{h}$的特征值小于1， 则梯度会呈次方衰减，称为**梯度消失**；若大于1，则梯度会呈次方增长，称为**梯度爆炸**

- 梯度消失会导致来自远处单词的梯度难以传递到当前单词，即模型“遗忘的速度过快”，通常有效的传播长度只有7个单词左右。
- 梯度爆炸会导致模型难以收敛

#### 处理梯度爆炸——gradient clipping

如果得到的梯度太大，则将其缩放到长度阈值。

#### 处理梯度消失——Long Short-Term Memory RNNs (LSTM)

在原有的RNN基础上加上**记忆层 (Cell Layer, c)**，并通过**门 (gates)** 来控制记忆的存留，更新与使用。

![[Pasted image 20250728215456.png|700]]

- **forget gate (f)**：控制上一次的记忆有多少被存留
$$
f^{(t)} = \sigma(W_{f}h^{(t - 1)} + U_{f} x^{(t)} + b_{f})
$$
- **input gate (i)**：控制新的记忆有多少被写入
$$
i^{(t)} = \sigma (W_{i}h^{(t - 1)} + U_{i} x^{(t)} + b_{i})
$$
- **output gate (o)**：控制有多少记忆被用于预测当前单词
$$
o^{(t)} = \sigma(W_{o}h^{(t-1)} + U_{o}x^{(t)} + b_{o})
$$

对于每一个输入单词，更新流程如下：

1. 产生新的记忆

$$
\tilde{c}^{(t)} = \tanh (W_{c} h^{(t-1)} + U_{c}x^{(t)} + b_{c})
$$

2. 根据新的记忆更新原有记忆

$$
c^{(t)} = f^{(t)} \circ c^{(t-1)} + i^{(t)} \circ \tilde{c}^{(t)}
$$

3. 根据更新后的记忆产生当前输出

$$
h^{(t)} = o^{(t)} \circ \tanh c^{(t)}
$$
4. 之后按照RNN的方式处理即可