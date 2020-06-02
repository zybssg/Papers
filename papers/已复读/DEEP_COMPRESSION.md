# Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding

ICLR 2016

Song Han,Huizi Mao,William J. Dally

[论文链接](https://arxiv.org/abs/1510.00149)

 

 

## abstract

Neural networks are both computationally intensive and memory intensive, making them difficult to deploy on embedded systems with limited hardware resources. To address this limitation, we introduce “deep compression”, a three stage pipeline:pruning, trained quantization and Huffman coding, that work together to reduce the storage requirement of neural networks by 35x to 49x without affecting their accuracy. Our method first prunes the network by learning only the important connections. Next, we quantize the weights to enforce weight sharing, finally, we apply Huffman coding. After the first two steps we retrain the network to fine tune the remaining connections and the quantized centroids. Pruning, reduces the number of connections by 9x to 13x; Quantization then reduces the number of bits that represent each connection from 32 to 5. On the ImageNet dataset, our method reduced the storage required by AlexNet by 35x, from 240MB to 6.9MB, without loss of accuracy. Our method reduced the size of VGG-16 by 49x from 552MB to 11.3MB, again with no loss of accuracy. This allows fitting the model into on-chip SRAM cache rather than off-chip DRAM memory. Our compression method also facilitates the use of complex neural networks in mobile applications where application size and download bandwidth are constrained. Benchmarked on CPU, GPU and mobile GPU, compressed network has 3x to 4x layerwise speedup and 3x to 7x better energy efficiency.

 

神经网络既需要密集的计算，又需要大量的内存，这使得它们很难在硬件资源有限的嵌入式系统上部署。为了解决这个限制，我们引入了深度压缩，一个三阶段的管道:修剪，训练量化和霍夫曼编码，共同工作，减少神经网络的存储需求35至49，而不影响其准确性。我们的方法首先通过只学习重要的连接来删除网络。然后对权重进行量化，实现权重共享，最后应用霍夫曼编码。在前两个步骤之后，我们对网络进行再培训，以对剩余的连接和量化中心进行微调。修剪，减少连接数9到13;量化然后减少代表每个连接的比特数从32到5。在ImageNet数据集上，我们的方法将AlexNet所需的存储空间减少了35个，从240MB减少到6.9MB，并且没有损失准确性。我们的方法将VGG-16的大小减少了49，从552MB减少到11.3MB，同样没有损失准确性。这允许将模型拟合到片内SRAM缓存而不是片外DRAM内存中。我们的压缩方法还有助于在应用程序大小和下载带宽受限的移动应用程序中使用复杂的神经网络。以CPU、GPU和移动GPU为基准，压缩网络具有3 - 4层加速和3 - 7层能效提升。

 

## model structure

![计算机生成了可选文字: Pruning: less number Of weights  Quantization: less blts per weight  Cluster the Weights  Huffman Encoding  Encode Weights  Encode Index  onginal  network ，  origlnal ，  Train Connectivity  Prune Connections  Train Weights  accuracy |  0  27x Ix  reductlon I  accuracy  0  ， 35x 9x  I reductlon  丨 |  0  丨 9x ． 13x  丨 reduction ，  Generate Code Book  Quantize the Weigh  with Code Book  Retrain Code Book  Figure l: The three stage compression pipeline ： pruning, quantization and Huffman coding. Pruning  reduces the number of weights by 10 × ， while quantization further improves the compression rate ：  between 27 × and 31 × ． Huffman coding gives more compression: between 35 × and 49 × ． The  compression rate already included the meta-data for sparse representation. The compression scheme  doesn ， t incur any" accuracy loss. ](file:///C:/Users/zyb/AppData/Local/Temp/msohtmlclip1/01/clip_image001.png)

本文提出的模型压缩方法，包含三个步骤：剪枝pruning，量化quantization，霍夫曼编码Huffman coding

###     剪枝pruning：

1. 1. 通过删除冗余连接来修剪网络，只保留信息量最大的连接。
   2. 首先通过正常的网络训练来学习整个网络。
   3. 接下来，修剪小权重连接：权重低于阈值的所有连接将从网络中删除。
   4. 最后，对网络进行再训练，以学习剩余稀疏连接的最终权重
   5. 步骤cd循环迭代

2. ### 量化quantization：

3. 1. 权重被量化，多个连接共享相同的权重，只需要存储有效权重和对应的索引

   2. compressed      sparse row (CSR) or compressed sparse column (CSC) format

   3. 1. 对于nxn的稀疏矩阵，并不需要nxn个数字来进行存储，可以采用CSR或者CSC方式进行存储，只需要2a+n+1个数字。其中a为非零元素的个数，n为行(CSR)或列(CSC)的数量
      2. 例如：

![img](file:///C:/Users/zyb/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)

其CSR格式由三个数组定义为：

![img](file:///C:/Users/zyb/AppData/Local/Temp/msohtmlclip1/01/clip_image003.png)

val数组：大小为矩阵A的非零元素的个数，保存矩阵A的非零元素（按从上往下，从左往右的行遍历方式访问元素）。

col_ind数组：和val数组一样，大小为矩阵A的非零元素的个数，保存val数组中元素的列索引。

row_ptr数组，大小为矩阵A的行数，保存矩阵A的每行第一个非零元素在val中的索引。

按照惯例，一般定义row_ptr[n+1] = nnz + 1,而nnz是A中非零元素的个数。

1. 本文在CSR和CSC的基础上进行改进，不存储绝对索引，而是相对索引。

![计算机生成了可选文字: Span Exceeds 8 = 2 ^ 3  11  13  Filler Zero ](file:///C:/Users/zyb/AppData/Local/Temp/msohtmlclip1/01/clip_image004.png)

 

1. 对于卷积层的参数的索引使用8位进行编码，对于全连接层参数的索引使用5位进行编码，当相对索引超过最大位时，用0进行填充，并置其相对索引为最大位

1. 量化与权重共享

![计算机生成了可选文字: weights  （ 32 bit float)  0.0  之 ，  0m2  0.0 ，  cluster  0  cluster index  （ 2 bit uint)  ， 2 0m2  0.0 ， ． 0 ． 02  centroids  0m0  reduce  -0.0  fine-tuned  centroids  .09  0 ． 05  .0  ． 98  ， ． 92  0  ， ． 53  Ir  0.0  .0  ． 03  group by 003  0m2  -0.0  0.04  gradient  -0.01 0 ． 03  0.0 ， -0.02  0m2 0.04 ](file:///C:/Users/zyb/AppData/Local/Temp/msohtmlclip1/01/clip_image005.png)

1. 对于一个输入size为4，输出size为4的全连接层，其参数矩阵为4x4，如左上所示。
2. 将16个参数进行分组（k-means)，图中分为四组，取每组的均值作为共享权值；
3. 新增一个4x4的索引矩阵，存储每个对应位置的组编号；
4. 原来4x4=16个32位的参数矩阵，则可用一个4x4的索引矩阵和一个4x1个32位的共享参数向量表示。
5. 假设量化后共有k个组，则索引矩阵中的元素均可使用位进行表示，如k=4时，组编号最大为3，可用11进行表示。![img](file:///C:/Users/zyb/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png)
6. 量化前需要16x32位进行存储，量化后需要16*+4*32位进行存储，压缩比为3.2倍

### 权值共享：k-means

1. 1. k-means算法的初始质心对聚类结果影响很大，考虑三种初始化方法：随机初始化，基于密度初始化，线性初始化

![计算机生成了可选文字: tDF  PDF  · · · density i n itia | i zat ion  ． ． ． InltlalizatlOn  0 0 0 random i nlt allzation  目 0 ． 2  005  weight v ](file:///C:/Users/zyb/AppData/Local/Temp/msohtmlclip1/01/clip_image008.png)

1. 权值值分布的PDF和CDF如图所示，其分布有两个波峰。随机初始化的初始化结果较为聚集在波峰位置，基于密度的初始化稍微分散一些，但也有聚集的特点，线性初始化完全是现行的，没有聚集的特点。
2. 实际中，大权重在网络中的作用非常大，但是大权重属于较为少的，因此随机初始化和基于密度初始化很少能够产生权重值较大的初始质心，而线性初始化不存在这个问题。实际实验中，也是线性初始化的效果最好。

1. 量化与参数共享的反向传播

2. 1. 如上上图所示，左下角为各个权值的梯度，进行反向传播时，对同一组参数的梯度也进行分组并求和得到组梯度，再利用组梯度对共享参数进行更新。

 

### 霍夫曼编码Huffman coding

1. 1. 用霍夫曼编码来利用有效权值的有偏分布。
   2. 权重索引和稀疏矩阵定位索引（相对索引）的分布也是非均衡的，因此使用Huffman      coding可以大大减小存储空间。

 

![计算机生成了可选文字: 1 m000  75000  25000  0  27  29  31  220000  165000  目 110000  0  55000  0  3  5  7 9 11 13 15 17 19 21 23 25 27  Sparse Matrix Location Index (Max Diff is 32 ）  29  31  3  5  7  9 11 13 15 17 19 21 23 25  Weight Index （ 32 Effective Weights) ](file:///C:/Users/zyb/AppData/Local/Temp/msohtmlclip1/01/clip_image009.png)

1. 实验表明,霍夫曼编码这些非均匀分布值节省20%-30%的存储空间。



 