#### 将passwords.py保存到根目录下

#### 运行python server.py

正常的时候分类为0，可能有问题的时候分类为1

#### 问题总结1：

首先是数据量，beijing2017年3290，18年123，chongqing2017年8590，感觉都不够

算条件概率的时候，直接数的话数目太少，不可能准确，用朴素贝叶斯的时候，发现概率都能超过1，而且也不一定是
单调的，importance不保证单调性

随机森林总是过拟合了，训练集上当然是接近100%,测试的时候就基本全是判断为0(正常)，所以基本上是80%的正确率，袋外估计的结果也是这样的