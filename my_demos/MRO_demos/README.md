# Python MRO (Method Resolution Order) 示例

## 介绍
本项目包含多个 Python 示例，旨在研究 Python 中的 **MRO (Method Resolution Order, 方法解析顺序)**。MRO 决定了在多重继承中，Python 解析方法调用的顺序，确保方法不会被重复调用，并遵循 **C3 线性化算法**。

## MRO 概念
在 Python 中，MRO 规则主要体现在 `super()` 调用时的继承关系顺序，具体表现为：
- Python 使用 **C3 线性化**（C3 Linearization）算法来计算 MRO。
- `super()` 按 MRO 顺序查找方法，避免重复访问基类。
- `Class.__mro__` 属性可以查看某个类的 MRO 解析顺序。