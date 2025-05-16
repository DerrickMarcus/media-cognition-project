# 媒体与认知 上机作业

## 整体框架

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

## 实验过程与结果

我原先使用的 Text Encoder 是 LSTM 模型，使用的是默认的 `batch_size` 等配置，但是训练的结果中，Recall@1 仅有约 2%，Recall@10 仅有约 6%。我先是尝试增大 `batch_size` 至较大值，例如128，使得每一批次读取的数据增多，增大图文对比学习的数据量，Recall@K 指标有了明显的提升：

```text
LSTM, batch_size=128, epochs=40

📈 Text → Image Retrieval:
Recall@1: 6.92%
Recall@5: 20.19%
Recall@10: 30.45%

📈 Image → Text Retrieval:
Recall@1: 6.85%
Recall@5: 20.61%
Recall@10: 31.39%
```

![202505161435335](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161435335.png)

接下来进一步增大 `batch_size` 至256，并训练50个 epoch，结果如下，此时 Recall@10 已经超过了50%：

```text
LSTM, batch_size=256, epochs=50

📈 Text → Image Retrieval:
Recall@1: 26.32%
Recall@5: 50.12%
Recall@10: 59.47%

📈 Image → Text Retrieval:
Recall@1: 26.32%
Recall@5: 47.01%
Recall@10: 56.08%
```

然后我尝试使用 Transformer 模型作为 Text Encoder，并再次增大 `batch_size` 至512，并更改 `embed_dim` 为512，嵌入维度增大，或许能够学习到更多细节特征，准确率有进一步提高。

```text
Tranformer, batch_size=512, embed_dim=512, epochs=30

📈 Text → Image Retrieval:
Recall@1: 27.11%
Recall@5: 51.71%
Recall@10: 60.75%

📈 Image → Text Retrieval:
Recall@1: 27.95%
Recall@5: 49.58%
Recall@10: 58.18%
```

![202505161453478](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161453478.png)

但是在这种情况下训练很快就出现过拟合，仅仅经过30个 epoch 后，验证集上的损失就开始出现略微的上升趋势，在第50个 epoch 时出现明显的过拟合。因此我将 `embed_dim` 降低至256，`batch_size` 仍为512，此时训练经过40个 epoch 后，验证集上的损失逐渐收敛并保持平稳，结果中的 Recall@K 指标也保持较高，且不低于 `embed_dim=512` 的情况。

(1) Transformer, batch_size=512, embed_dim=256, epochs=30:

![202505161455050](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161455050.png)

```text
Final Test Loss: 2.2788.

📈 Text → Image Retrieval:
Recall@1: 29.71%
Recall@5: 54.72%
Recall@10: 63.27%

📈 Image → Text Retrieval:
Recall@1: 29.81%
Recall@5: 51.90%
Recall@10: 60.87%
```

(2) Transformer, batch_size=512, embed_dim=256, epochs=40:

![202505161455639](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161455639.png)

```text
Final Test Loss: 2.3385.

📈 Text → Image Retrieval:
Recall@1: 28.32%
Recall@5: 53.53%
Recall@10: 63.00%

📈 Image → Text Retrieval:
Recall@1: 29.16%
Recall@5: 52.22%
Recall@10: 61.10%
```

此时 Top1 Accuracy/Recall@1 接近30%，Recall@10 超过60%。当然，由于我追求快速改进模型的性能，上述调整超参数的过程中我没有控制单次只改变一个参数，虽然结果确实在改进，但是调整方法不够严谨，无法说明是哪一个参数的变化起到主导作用。LSTM 模型在时间迭代过程中天然地保持有序，不需要像 Transformer 中添加额外的位置编码，并且参数量更小，更加轻量，适用于较小语料下的训练。而 Transformer 模型通过全局自注意力，可以捕获长距离依赖，适用于较长文本和较大规模语料下的训练。而在本次实验中，由于数据集规模较小，总 captions 只有约40000条，且我们构建的序列长度仅为32，因此在这种小规模训练集上 LSTM 与 Transformer 性能接近，但是后续考虑到可以使用更大规模的训练集（Flickr30k 或 MS CoCo），我先使用了 Transformer 模型。

虽然使用不同的 TextEncoder，或者设置不同的 `batch_size` 和 `embed_dim`，最终训练集、验证集、测试集都较为接近，而 Top1 Accuracy/Recall@K 等检索指标有差异。原因可能是这些检索指标对嵌入向量的微小差异较为敏感，由于是检测与输入图片或者文本对应度最高的那几个样本，因此只关注相似度排名最高的几个样本，而损失函数是对全局样本做对齐，关注平均对齐效果，每一对正样本的相似度都要纳入损失函数。损失函数中，`batch_size` 决定了每个批次中负样本的个数，即使是在损失相同的情况下，`batch_size` 越大，每条正样本要和更多的负样本做对比，模型学到的界面更“紧凑”，检索时更容易将相似度高的样本排在前面。另外，嵌入维度 `embed_dim` 越大，可以编码更丰富的语义细节（但是可能出现过拟合，就像上面的结果中那样，后续可以使用图像增强和文本增强防止），在整体的损失接近下，嵌入向量之间两两的相似度也可能有较大差异。因此即使看似损失接近，检索指标也有差异，这也是为什么我们在训练过程中选择最优检索 Top1 Accuracy/Recall@K 作为最优模型保存，而不是最小损失函数。

然后尝试更换使用 SGD 优化器：

```py
    optimizer = torch.optim.SGD(
        list(img_encoder.parameters()) + list(txt_encoder.parameters()),
        lr=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
```

结果为：

![202505161457446](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161457446.png)

```text
Transformer, batch_size=512, embed_dim=256, epochs=40, Optimizer=SGD:

Final Test Loss: 2.7569.
📈 Text → Image Retrieval:
Recall@1: 23.41%
Recall@5: 46.37%
Recall@10: 54.57%

📈 Image → Text Retrieval:
Recall@1: 24.67%
Recall@5: 44.51%
Recall@10: 53.81%
```

此时 SGD 优化器的结果略差于 Adam 优化器的结果，Recall@K 指标略有下降，而且同样是经过40个 epoch，使用 Adam 优化器时训练集上损失可以降低至0.2左右，验证集上损失可以降低至2.4左右，测试集损失约在2.3，而使用 SGD 优化器时，训练集上的损失降至0.4左右，验证集损失降至2.8左右，测试集损失约在2.75。当然，这也仅仅是一次的训练结果，SGD 优化器需要有比 Adam 优化器更高的学习率，因此我将学习率从 1e-4 调整至 1e-3，但不一定是最优的参数，后续可以再做探究和改进。

综合上述结果，我们最终选择 Transformer 模型作为 Text Encoder，使用 `batch_size=512` 和 `embed_dim=256` ，使用 Adam 优化器，训练40个 epoch，最终的 Recall@K 指标如下：

```text
Final Test Loss: 2.3385.

📈 Text → Image Retrieval:
Recall@1: 28.32%
Recall@5: 53.53%
Recall@10: 63.00%

📈 Image → Text Retrieval:
Recall@1: 29.16%
Recall@5: 52.22%
Recall@10: 61.10%
```

## 可视化展示

编写和运行可视化文件 `visualize.py` ，并随机从数据集中挑选几个 caption 作为输入，对输入文本进行检索。刚开始进行检索的时候发现会出现重复的情况，类似于这样：

![202505161610819](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161610819.png)

原因似乎是原本每张图片对应5条不同的 caption 描述，数据集中会有同一个图片的文件名出现5次，对应的 embedding 也会重复计算，因此检索结果中出现重复。因此我又添加了去重处理，对 `dataset.pairs` 中的文件名保留顺序地去重，拿到 `unique_fns` 作为文件名列表，结果如下：

(1) Caption: "One brown dog is bearing its teeth at another brown dog with a green collar in a park ."

![202505161621834](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161621834.png)

(2) Caption: "A girl with a short haircut and eyebrow piercing bites her finger and crosses her arms ."

![202505161625793](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161625793.png)

(3) Caption: "A snowboard rider jumping high on his snowboard in the snow ."

![202505161627537](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161627537.png)

对于这份查询，结果中的第1、2、4、5个似乎都比较符合，难以区分哪一个真正正确，可能是文本表述的 caption 中给出的信息太少、比较模糊和笼统、缺少一些独特的细节信息，或者是这几张图片对应的文本描述中有几个词重合度较高，例如 snowboard, jumping high, snow。我们使用的模型对文本和图片编码得到的嵌入向量，大多只包含整张图或整句文本的整体信息，在局部细节方面，比如滑板的具体朝向、背景是否有人、动作的特殊角度都被压缩掉了，这些样本在共享空间内本身就十分近似，而模型并没有对这方面专门进行区分处理。这也为后面改进模型提供了一个方向，让模型能够更好的区分与正样本十分接近的负样本，学习的更加精细。

(4) Caption: "A man in a white shirt walks in the tall grass holding a stick ."

![202505161638446](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161638446.png)

(5) Caption: "The boy in the black sweatshirt is hitting the yellow object held by the boy in the blue sweatshirt ."

![202505161645809](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161645809.png)

(6) Caption: "A man riding a bike wearing flannel shirt , plaid pants and a blue backpack while riding on a street near buildings and parked cars ."

![202505161647783](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161647783.png)

推测较长文本包含的信息更多（虽然可能被截断），与其他负样本区分度更高，因此结果更加准确。然而对于这个检索，第一个看似比较符合，实则并非正确结果，经过查找可知，正确图片应为：

![202505161651829](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161651829.jpg)

可见对于本次查询，前5个结果并没有正确答案，甚至前10个结果中都没有正确答案：

![202505161654096](https://cdn.jsdelivr.net/gh/DerrickMarcus/picgo-image/images/202505161654096.png)

可见，模型对于包含 bike 信息的图像学习效果不如不够理想，低于平均水平。

## 实验总结

刚开始接触多模态模型的时候，我以为是较为复杂的结构，但其实思路和流程很清晰，只是两个解耦的文本编码器和图像编码器，分别输出对文本和推向的嵌入向量，然后两两对比学习。在初步的模型搭建好之后，检索指标已经表现较好，Top1 Accuracy/Recall@1 接近30%，Recall@10 超过60%。但是后续还可以尝试对模型进行多方面的改进和优化，可以进一步改进模型结构、增加模型的复杂度，训练数据方面对数据集进行预处理，超参数方面尝试多种优化器和学习方法等。
