# 上海大学课程项目

## 自然语言处理 Natural Language Processing

## 第8组 图片描述识别  Image Captioning

### 项目概要

本项目旨在实现一个基于Transformer架构的图像描述生成模型（Image Captioning）。通过结合预训练的ResNet50模型作为图像编码器，Transformer作为解码器，模型能够自动生成自然语言描述，用以表达输入图像中的内容。该项目的主要目标是在图像输入的基础上生成高质量的自然语言描述。

### 主要特性

- **图像特征提取**：使用预训练的 ResNet50 模型对输入图像进行特征提取，生成高维度的特征向量。
- **基于 Transformer 的解码器**：通过 Transformer 解码器，将图像特征逐词生成相应的文本描述。
- **解码策略**：支持贪心解码（Greedy Decoding），提高生成描述的质量和多样性。
- **Flickr8k 数据集**：在 Flickr8k 数据集上进行训练和测试，以确保模型的泛化能力和准确性。
- **性能评估**：通过 BLEU-1、BLEU-4 和 ROUGE 分数评估模型生成的图像描述质量。

### 系统架构

1. **图像编码器**：采用预训练的 ResNet50 提取图像特征，将其转化为可供解码器处理的特征向量。
2. **文本解码器**：基于 Transformer 的解码器模型接收图像特征，并生成相应的自然语言描述。
3. **解码策略**：
   - **贪心解码**：每次选择概率最高的词，逐步生成完整的描述。

### 文件结构

- `gradio_interface.py`: 项目的gradio界面，入口。
- `dataloader.py`：负责加载 Flickr8k 数据集，并进行图像预处理与标签处理。
- `decoder.py`：包含基于 Transformer 的解码器，用于逐词生成图像描述。
- `decoding_utils.py`：包含贪心解码的实现。
- `inference.py`：用于推理阶段，通过加载模型和图片生成描述。
- `evaluation.py`：实现了 BLEU 和 ROUGE 评分函数，用于评估模型性能。
- `main.py`：训练模型的主脚本。
- `utils.py`：包含项目中的一些工具函数。
- `config.json`：模型参数和训练设置。

### 安装与运行

1. 按照 [machinelearningmastery 博客](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)中的说明,下载Flickr8k数据集并放入相应位置。
2. 下载 [GloVe 嵌入](https://nlp.stanford.edu/projects/glove/)，名称为 “glove.6B.zip ”。
3. 打开 Anaconda 提示符，并使用以下命令导航到该 repo 的目录： cd PATH_TO_THIS_REPO。
4. 执行 conda env create -f environment.yml 以建立一个包含所有必要依赖项的环境。
5. 执行： conda activate pytorch-image-captioning 激活之前创建的环境。
6. 在[配置文件](https://github.com/sakura0224/Group8-Image-Captioning/blob/main/config.json)中修改 glove_dir 项，将其更改为下载 GloVe 嵌入文件的目录路径。
7. 运行 python prepare_dataset.py。会执行以下步骤：
   - 加载图像的原始标题
   - 对标题进行预处理
   - 生成标题语料库中出现的词库
   - 为之前创建的词汇表中出现的标记提取 GloVe 嵌入
   - 进一步调整词库（舍弃没有嵌入词的词语）
   - 数据集拆分： 根据预定义的分割，分离图像-标题映射
8. 运行 python main.py 启动模型训练。

### 参考

[How to Develop a Deep Learning Photo Caption Generator from Scratch](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)
[senadkurtisi/pytorch-image-captioning](https://github.com/senadkurtisi/pytorch-image-captioning)
[sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
