---
title: "RRG Metric"    # 标题，去掉横短线病转换为标题格式
date: 2024-12-10T17:45:09+08:00                                               # 发布日期
tags: ["RRG","Metric"]                                                      # 分类和标记，用于过滤
author: "Jeff"                                                  # 作者
# author: ["Me", "You"] # multiple authors
showToc: true                                                   # 显示目录
TocOpen: false                                                  # 默认展开
draft: false                                                    # 是否为草稿（True则会发布）
hidemeta: false                                                 # 隐藏元信息（作者、发布日期等）
comments: false                                                 # 是否comments
description: ""                                                 # 文章描述
canonicalURL: "https://canonical.url/to/page"                   # idk
disableShare: false                                             # 禁止分享
disableHLJS: false                                              # 禁用代码高亮
hideSummary: false                                              # 隐藏文章摘要
searchHidden: false                                             # 在search里隐藏文章
ShowReadingTime: true                                           # 显示阅读时间
ShowBreadCrumbs: true                                           # 显示面包屑导航
ShowPostNavLinks: true                                          # 显示文章导航（下一篇，上一篇）
ShowWordCount: true                                             # 字数统计
ShowRssButtonInSectionTermList: true                            # idk
UseHugoToc: true 
math: true                                               # 使用Hugo生成的目录
# cover:
#     image: "<image path/url>" # image path/url
#     alt: "<alt text>" # alt text
#     caption: "<text>" # display caption under cover
#     relative: false # when using page bundles set this to true
#     hidden: true # only hide on current single page
editPost:
    URL: "https://github.com/Jeffzvz.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## 一、 BLEU

全称 Bilingual Evaluation Understudy，用于评估机器翻译和参考翻译的相似度。

-  n-gram匹配和精度
  
  评估机翻和参考翻译之间的重叠度。比如有如下两个句子：
  
  ```text
  C: I love pig and dog dog.
  R: I love dog.
  ```
  
  对于**1-gram**，如下所示，计算得precision为 3/6，3表示C中有3个word与R重叠 (去除了重复部分)，6表示C长度为6；
  
  ```text
  C: I; love; pig; and; dog; dog;
  R: I; love; dog;
  ```
  
  对于2-gram，如下所示，计算得precison为 1/5;
  
  ```text
  C: I love; love pig; pig and; and dog; dog dog;
  R: I love; love dog;
  ```

        但是n-gram毕竟只是表示局部词之间的重叠度，也不能表示语义信息，而 句子是可以从不同角度理解的，故它并不是一个很好的评估指标；



- BLEU
  
  公式表示为：
  
  $$
  BLEU=BP\times \text{exp}(\sum_{n=1}^Nw_n\text{log }p_n)
  $$
  
  其中BP是惩罚因子，$w_n$是每个n-gram的权重，通过为$\frac{1}{N}$，$p_n$则是不同n-gram的precision。而BP的公式如下：

  $$
  BP =
\begin{cases}
1, & \text{如果 } c > r \\
\exp\left(1 - \frac{r}{c}\right), & \text{如果 } c \leq r
\end{cases}
  $$
  
  BLEU-1就是只计算BP和1-gram； BLEU-2就是计算BP和1-gram以及2-gram；如此……同样，BLEU也是通过n-gram来计算的， **它也不具备理解语义信息，或者对句子词的顺序也不敏感，且缺乏对长句子的惩罚，重复单词也不能反映到得分上**。



## 二、ROUGE

全称Recall-Oriented Understudy for Gisting Evaluation， 专注于召回率的评估。

- 变体1 ROUGE-N

评估n-gram的重叠程度，不同的是分母从机器翻译的n-gram总数变成了参考文本的n-gram总数。公式如下：

$$
ROUGE-N = \frac{\text{机翻和参考文本匹配的n-gram}}{\text{参考文本的n-gram}}
$$



- 变体2 ROUGE-L （常用）

基于 最长公共子序列（Longest Common Subsequence, LCS) , LCS允许不连续匹配，但要保持顺序，更适合句子级别的匹配，公式如下：

$$
ROUGE-L = \frac{LCS(机翻，参考)}{参考文本长度}
$$

举个例子：

```
R: The cat is on the mat.
C: The cat mat mat mat on
```

那么LCS： ['The' , 'cat', 'mat'] = 3

```
R: The cat is on the mat.
C: The cat on mat mat mat.
```

则 LCS: ['The' , 'cat', 'on' , mat'] = 4

所以ROUGE本身的优势是一个是它更关注召回率（以参考文本为分母），第二个就是ROUGE-L注重句子中词语出现的顺序。但是也有缺陷，**一是同样不能理解语义信息，第二个就是没有办法处理重复词。**



## 三、METEOR

全称Metric for Evaluation of Translation with Explicit ORdering，计算过程如下：

1. 词匹配
   
   包括完全匹配，单复数匹配，同义词匹配（通过wordnet这样的词典寻找）、词形变化匹配。匹配的意思是dog和dogs、run和running、fast和quick这样的词都算是一样的

2. Precision和recall
   
   P = 匹配词汇 /  机翻总词汇
   
   R = 匹配词汇 / 参考总词汇

3.  BP计算
   
   $$
   BP = \gamma(\frac{\text{chunks}}{\text{unigram match}})^\theta
   $$
   
   其中Chunks的概念如下：
   
   ![](/RRG_1_picture_saved/2024-12-10-17-20-05-image.png)

4. METEOR计算
   
   $$
   \text{METEOR}=\text{F1}\times\text{BP}
   $$

其中$F1 = \frac{(\alpha^2+1)P}{R+\alpha P}$ 。METEOR的好处有几个，一是有词性和同义词匹配，而这是其他两个没有的。通过chunks惩罚，METEOR也考虑了次序和连续性，这是BLEU没有的；且其平衡了Precision和recall，更全面。但是也有几个缺陷，一个是基于n-gram，那必定会错过更高级的语义和句法差异；第二个计算复杂性高；第三个就是对外部库有依赖；第四个是有权重和超参的选择。

## Reference:
[1] https://zhuanlan.zhihu.com/p/659729027
