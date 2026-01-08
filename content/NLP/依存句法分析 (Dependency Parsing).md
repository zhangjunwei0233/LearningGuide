---
title: 依存句法分析 (Dependency Parsing)
draft: "false"
tags:
  - NLP
---
## 句子的语言结构

### 观点1：Phrase Structure与CFG (Context-free Grammar)

将一个句子表示为一棵树，树的结构反应短语与短语的嵌套结构

一个CFG的生成规则示例如下：
```tex
NounPrase -> Det (Adj)* N
NounPrase -> NounPrase PrepositionPrase
PrepositionPrase -> Prep NounPrase

Det -> {a, the, ...}
Adj -> {large, small, ...}
...
```

### 观点2：Dependency structure (更常用)

将一个句子表示为一棵树，树的结构反应单词与单词之间的依赖关系

![[Pasted image 20250726111310.png|700]]

Dependency structure允许non-projective parsing (及存在交叉依赖，如图中的红线)。但是使用CFG构建的句子结构不可能出现这类情况。


## Dependency Parsing

给定一个句子，有几种方式可以将其解析为Dependency structure

### transition-based parsing

遍历句子中的词，过程中维护一个栈，每次选择以下三种操作 (Transition) 之一，直至完全遍历且栈中只剩下ROOT一个元素。

1. SHIFT：将Buffer中最左侧的词入栈
2. LEFT-ARC：将栈顶左侧的词标记为栈顶的dependency，并将该词移除
3. RIGHT-ARC：将栈顶的元素标记为其左侧元素的dependency，并将该词移除

![[Pasted image 20250728090149.png|700]]

| Stack                                         | Buffer                                                   | New dependency          | Transition     |
| --------------------------------------------- | -------------------------------------------------------- | ----------------------- | -------------- |
| \[ROOT\]                                      | \[I, presented, my, findings, at, the, NLP, conference\] |                         | Initial Config |
| \[ROOT, I\]                                   | \[presented, my, findings, at, the, NLP, conference\]    |                         | SHIFT          |
| \[ROOT, I, presented]                         | \[my, findings, at, the, NLP, conference\]               |                         | SHIFT          |
| \[ROOT, presented\]                           | \[my, findings, at, the, NLP, conference\]               | presented -> I          | LEFT-ARC       |
| \[ROOT, presented, my\]                       | \[findings, at, the, NLP, conference\]                   |                         | SHIFT          |
| \[ROOT, presented, my, findings\]             | \[at, the, NLP, conference\]                             |                         | SHIFT          |
| \[ROOT, presented, findings\]                 | \[at, the, NLP, conference\]                             | findings -> my          | LEFT-ARC       |
| \[ROOT, presented\]                           | \[at, the, NLP, conference\]                             | presented -> findings   | RIGHT-ARC      |
| \[ROOT, presented, at\]                       | \[the, NLP, conference\]                                 |                         | SHIFT          |
| \[ROOT, presented, at, the\]                  | \[NLP, conference\]                                      |                         | SHIFT          |
| \[ROOT, presented, at, the, NLP\]             | \[conference\]                                           |                         | SHIFT          |
| \[ROOT, presented, at, the, NLP, conference\] | \[\]                                                     |                         | SHIFT          |
| \[ROOT, presented, at, the, conference\]      | \[\]                                                     | conference -> NLP       | LEFT-ARC       |
| \[ROOT, presented, at, conference\]           | \[\]                                                     | conference -> the       | LEFT-ARC       |
| \[ROOT, presented, conference\]               | \[\]                                                     | conference -> at        | LEFT-ARC       |
| \[ROOT, presented\]                           | \[\]                                                     | presented -> conference | RIGHT-ARC      |
| \[ROOT\]                                      | \[\]                                                     | ROOT -> presented       | RIGHT-ARC      |

可以训练一个神经网络，在每一步作出正确的Transition决策：

![[Pasted image 20250726115013.png|700]]

### graph-based parsing

对于句子中的每一个词，计算其是其他所有词的依赖的可能性。
之后使用诸如最小生成树的方式建树。