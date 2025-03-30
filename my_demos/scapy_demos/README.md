# spaCy Demos

## 介绍

本项目包含多个 `spaCy` 的示例，`spaCy` 是一个高性能的自然语言处理（NLP）库，支持 **分词、词性标注、实体识别、依存句法分析**
等功能。

## spaCy 主要功能

- **分词（Tokenization）**：将文本拆分为单词、短语或子词单元。
- **词性标注（POS Tagging）**：识别单词的词性，如名词、动词、形容词等。
- **命名实体识别（NER）**：识别文本中的专有名词，如人名、地名、公司名称等。
- **依存句法分析（Dependency Parsing）**：分析单词之间的句法关系。
- **词向量（Word Vectors）**：支持预训练的词向量模型（如 `en_core_web_md`、`en_core_web_lg`）。

## 安装 spaCy

1. 在运行示例代码之前，需要安装 `spaCy`：

```bash
pip install spacy
```

2. 下载预训练语言模型

- spaCy 需要加载语言模型来进行文本处理。可以选择不同大小的模型：

```bash
python -m spacy download en_core_web_sm  # 小型模型（速度快）
python -m spacy download en_core_web_md  # 中型模型（包含词向量）
python -m spacy download en_core_web_lg  # 大型模型（更好的语义理解）

# 如果已经下载 .whl 文件，可以用以下命令手动安装：
pip install en_core_web_sm-3.8.0-py3-none-any.whl
```
