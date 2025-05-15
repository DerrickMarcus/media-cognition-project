# 媒体与认知 上机作业

## 整体方案

本次上机实验的目标是搭建一个简单的多模态模型，通过图像和文本的对比学习，实现图像和文本之间的相互检索。

模型的主题是一个图像编码器 Image Encoder 和一个文本编码器 Text Encoder，模型所学习的数据集是一组图片和若干条对与图像信息的文本描述 caption，Image Encoder 和 Text Encoder 分别对输入的图像和文本进行编码，得到图像和文本的特征向量 Embedding，并且是相同维度。在共享空间中，图像嵌入和文本嵌入两两之间计算余弦相似度，得到图像和文本之间的相似度矩阵。训练过程中，通过相似度矩阵计算 InfoNCE 损失函数，并自动反向传播更新模型参数，由此完成模型的训练。

### 准备部分

文件 `utils.py` 定义了 `SimpleTokenizer` 类，主要功能为：

1. `self.build_vocab` ：根据若干条文本 `captions` 构建词表 `self.word2idx` ，将数据集中出现的较高频词映射到整数索引。填充 PAD 的定义其索引为0，词表中未出现的词定义其索引为1。
2. `self.encode` ：将给定输入文本 `text` 中的每个词转换为其在词表中的索引，返回一个整数列表 `token_ids` ，并截取或填充到固定长度，便于不同序列拼接为张量。
3. `self.__len__` ：返回词表的总长度。

文件 `data_loader.py` 定义了 `Flickr8kDataset` 数据集类，主要功能为：

1. `self._load_pairs` ：从文件提取图片文件名和对应的文字描述，生成一个“图像文件名+文本描述”的列表。
2. `self.__len__` ：返回数据集的大小。
3. `self.__getitem__` ：根据索引，返回图像张量，形状为 `(B, 3, H, W)` （彩色图像通达数为3），以及对应的文本描述经过 encode 之后的 ID 张量，形状为 `(B, max_len)` 。

文件 `loss.py` 定义了 InfoNCE 损失函数，根据图像的嵌入张量和文本的嵌入张量，计算余弦相似度矩阵，然后利用交叉熵得到“图像到文本方向的损失函数”和“文本到图像方向的损失函数”，取平均值作为最终的损失函数。

### 模型部分

图像编码器 `ImageEncoder` 类：使用自定义的 `ResNet18` 模型。在模型的前向传播中，输入的图像张量依次经过卷积、池化，并在4个 layer 阶段中通过下采样 downsample ，使得特征图大小减半、通道数翻倍，最后通过全局平均池化、展平得到一维向量。通过全连接层，将 ResNet 的输出映射到共享空间，得到归一化的图像的嵌入张量 embedding。另外，通过残差连接，可以让模型学习“残差为0”，而不是恒等映射，可以加快模型的收敛性速度，缓解梯度消失问题。

文本编码器 `TextEncoder` 类：使用自定义的 Transformer 模型。首先在 `PositionalEncoding` 中对输入的文本嵌入进行位置编码，保证无序的子注意力能够区分 token 的先后顺序。在 `MultiHeadSelfAttention` 多头自注意力中，先一次性生成投影 Q,K,V 矩阵，划分为多个头并行计算，分别计算注意力分数和注意力权重（缩放点积 + softmax），最后合并多头，输出投影结果。在 `TransformerBlock` 中，子注意力子层和前馈网络子层均使残差连接，可以增大网络深度。最后在 `TransformerTextEncoder` 的前向传播中，输入的文本张量依次经过嵌入、位置编码、残差连接的 layer 层和全连接层，得到归一化的文本的嵌入张量 embedding。

## 实验过程

进行训练时，

实验过程中，我先是尝试增大 `batch_size` 至较大值，例如512，尝试增大嵌入向量的维度 `embed_dim` 至512，这样每一批次读取的数据增多，嵌入维度增大，或许能够学习到更多细节特征，准确率会有一定提高。但是在这种情况下训练很快就出现过拟合，仅仅经过30个 epoch 后，验证集上的损失就开始出现略微的上升趋势，在第50个 epoch 时出现明显的过拟合。因此我将 `embed_dim` 降低至256，`batch_size` 仍为 512，此时训练经过 40 个 epoch 后，验证集上的损失逐渐收敛并保持平稳，结果中的 Top@K 指标也保持较高，且不低于很大 `embed_dim` 的情况。

## 结果分析

## 可视化展示

编写可视化文件 `visualize.py` 。

## 实验总结

使用 LSTM 模型，batch_size=128,epochs=40测试及测试结果：

```text
📈 Text → Image Retrieval:
Recall@1: 6.92%
Recall@5: 20.19%
Recall@10: 30.45%

📈 Image → Text Retrieval:
Recall@1: 6.85%
Recall@5: 20.61%
Recall@10: 31.39%
```

使用 LSTM 模型，batch_size=256,epochs=50测试及测试结果：

```text
📈 Text → Image Retrieval:
Recall@1: 26.32%
Recall@5: 50.12%
Recall@10: 59.47%

📈 Image → Text Retrieval:
Recall@1: 26.32%
Recall@5: 47.01%
Recall@10: 56.08%
```

使用 transformer 模型，batch_size=512,num_workers=12,embed_dim=512,epochs=30测试及测试结果：

```text
📈 Text → Image Retrieval:
Recall@1: 27.11%
Recall@5: 51.71%
Recall@10: 60.75%

📈 Image → Text Retrieval:
Recall@1: 27.95%
Recall@5: 49.58%
Recall@10: 58.18%
```

使用 transformer 模型，batch_size=512,num_workers=12,embed_dim=256,epochs=30测试及测试结果：
Final Test Loss: 2.2788.

📈 Text → Image Retrieval:
Recall@1: 29.71%
Recall@5: 54.72%
Recall@10: 63.27%

📈 Image → Text Retrieval:
Recall@1: 29.81%
Recall@5: 51.90%
Recall@10: 60.87%

使用 transformer 模型，batch_size=512,num_workers=12,embed_dim=256,epochs=40测试及测试结果：
Final Test Loss: 2.3385.

📈 Text → Image Retrieval:
Recall@1: 28.32%
Recall@5: 53.53%
Recall@10: 63.00%

📈 Image → Text Retrieval:
Recall@1: 29.16%
Recall@5: 52.22%
Recall@10: 61.10%

换一个优化器
    optimizer = torch.optim.SGD(
        list(img_encoder.parameters()) + list(txt_encoder.parameters()),
        lr=1e-3,  # 通常 SGD 要比 Adam 用稍大的 lr
        momentum=0.9,  # 动量
        weight_decay=1e-4,  # 权重衰减
        nesterov=True,  # 可选：Nesterov 动量
    )

Final Test Loss: 2.7569.
📈 Text → Image Retrieval:
Recall@1: 23.41%
Recall@5: 46.37%
Recall@10: 54.57%

📈 Image → Text Retrieval:
Recall@1: 24.67%
Recall@5: 44.51%
Recall@10: 53.81%
