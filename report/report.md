# 媒体与认知 上机作业

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

使用 LSTM 模型，batch_size=512,num_workers=12,embed_dim=512,epochs=30测试及测试结果：

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
