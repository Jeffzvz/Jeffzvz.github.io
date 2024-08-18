---
title: "🍇IoU系列学习"    # 标题，去掉横短线病转换为标题格式
date: 2024-08-18T17:13:23+08:00                                             # 发布日期
Tags: ["Object Detection","Evaluation Matric"]                                     # 分类和标记，用于过滤
author: "zhm"                                                  # 作者
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
UseHugoToc: true                                                # 使用Hugo生成的目录
math : true
editPost:
    URL: "https://github.com/Jeffzvz.github.io/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## 1 IoU

$$
\begin{aligned}
\text{IoU}(box1,box2) &= \frac{box1\cap box2}{box1\cup box2} \\
\text{IoU Loss} &= 1 - \text{IoU}
\end{aligned}
$$

**🤢缺点**

- Dt与Gt不相交时, IoU为0，损失为1，既即使两个boxes相距很远，Loss仍为1，直觉上来说，相距越远损失应该更大。如下图所示, $\text{IoU Loss(box1,box2)}=\text{IoU Loss(box1,box3)}=1$

![](iou_/2024-08-18-16-27-10-image.png)

- IoU不能反映2个boxes的重合度大小，只能反映重叠程度，比如下面三幅图的IoU都是相同的，但明显左图回归最好，而右图回归很差。

![](iou_/2024-08-18-16-53-57-image.png)

**🥸 代码**

- 思路
  
  对于intersection面积的计算： 找到最大的左上角坐标，以及最小的右下角坐标，相乘即可。
  
  对于union面积的计算：计算box1和box2的面积，面积之和减去intersection即可。
  
  运用公式$\text{IoU} =\frac{\text{intersection}}{\text{union}}$ 即可

- 官方库函数 `torchvision.ops.box_iou(box1,box2)`
  
  其中传入的box尺寸为 [N,4]

```python
import torch
import torchvision.ops as ops

def calculate_iou(box1: torch.tensor, box2: torch.tensor):
    inter_x1 = torch.max(box1[:,0],box2[:,0])
    inter_y1 = torch.max(box1[:,1],box2[:,1])
    inter_x2 = torch.min(box1[:,2],box2[:,2])
    inter_y2 = torch.min(box1[:,3],box2[:,3])
    inter_width = torch.clamp(inter_x2-inter_x1 , min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1,min=0)

    box1_area = (box1[:,2]-box1[:,0])*(box1[:,3]-box1[:,1])
    box2_area = (box2[:,2]-box2[:,0])*(box2[:,3]-box2[:,1])
    intersection = inter_height*inter_width
    union = box1_area + box2_area - intersection
    iou = intersection / union
    return iou, intersection, union

box1 = torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(0)
box2 = torch.tensor([2.0, 3.0, 4.0, 5.0]).unsqueeze(0)

iou,_,_ = calculate_iou(box1,box2)
iou_ops = ops.box_iou(box1,box2).item()
print(f'Calculate iou {iou}')
print(f'Torchvision calculate iou {iou_ops}')  

#Calculate iou tensor([0.1429])
#Torchvision calculate iou 0.1428571492433548
```

## 2 GIoU

$$
\begin{aligned}
\text{GIoU} &= \text{IoU} - \frac{ac - u}{ac}\\
\text{GIoU Loss} &= 1- \text{GIoU}
\end{aligned}
$$

其中ac表示2个边界框的最小包络区域，如下图所示，浅黄色为box1和box3的最小包络区域，浅橘色为box2和box3的最小包络区域。

![](/iou_picture_saved/2024-08-18-16-31-29-image.png)

**😒改进点和说明**

- 改进1：解决了$\text{IoU}=0$时不能反映两个边界框远近的问题。 

当$\text{IoU}=0$时，$\text{GIoU}=-\frac{ac-u}{ac}$，其中$u=\text{box}_1\cup\text{box}_2$表示两个边界框的并集面积是不会变的，而ac是最小包络区域，当两个boxes间隔的越远，ac就越大，那么$\text{GIoU}\to -1$，则损失也就接近于2 

- 改进2：解决了重合度问题。

如图2所示，三幅图的IoU相同，但是GIoU分别为0.33, 0.25, -0.1。所以GIoU能更好地反映两者的重合度。

- $-1\le\text{GIoU}\le1$
  
  GIoU是IoU的下界，当两个boxes完全重合时，$ac=u$ 且 $\text{IoU}=1$，所以$\text{GIoU}=\text{IoU}=1$，损失为0
  
  在两个boxes完全不重合且无限远时，$\text{GIoU}=-1$

**🥸 代码**

- 思路
  
  计算IoU的思路如上所述
  
  计算ac面积：找到最小包络区域的左上角坐标（2个boxes的左上角坐标最小值），以及右下角坐标（2个boxes的右下角坐标的最大值），既可计算ac的面积
  
  同时从计算IoU的函数中返回union，按照如下公式计算GIoU即可：$\text{GIoU}=\text{IoU}-\frac{ac-u}{ac}$

- 官方库函数`torchvision.ops.generalized_box_iou(box1,box2)`

```python
def calculate_giou(box1: torch.tensor, box2: torch.tensor):
    iou,intersection,union = calculate_iou(box1,box2)
    ac_x1 = torch.min(box1[:,0],box2[:,0])
    ac_y1 = torch.min(box1[:,1],box2[:,1])
    ac_x2 = torch.max(box1[:,2],box2[:,2])
    ac_y2 = torch.max(box1[:,3],box2[:,3])
    ac = (ac_x2-ac_x1) * (ac_y2-ac_y1)
    giou = iou - torch.abs(ac-union)/ac
    return giou

box1 = torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(0)
box2 = torch.tensor([2.0, 3.0, 4.0, 5.0]).unsqueeze(0)

giou = calculate_giou(box1,box2)
giou_ops = ops.generalized_box_iou(box1,box2)
print(f'Calculate giou {giou}')
print(f'Torchvision calculate giou {giou_ops}')  

#Calculate giou tensor([-0.0794])
#Torchvision calculate giou tensor([[-0.0794]])
#
```
