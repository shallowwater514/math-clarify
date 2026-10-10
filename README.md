# Math Clarify（中文）

一个离线运行的数学公式分析原型：规范化常见公式写法、识别需要澄清的歧义、进行公式词项检索，并可选用 LaTeX→SymPy 解析符号结构。

**状态：v0.1.0 Alpha。** 目前适合研究、教学实验和离线评测。它不会证明两个公式等价，也不能仅凭 `|A|` 判断是绝对值、行列式、向量模还是集合基数。遇到这种情况，它会返回可向用户提出的问题。

English documentation: [README_EN.md](README_EN.md)

## 安装

克隆仓库后，在仓库根目录运行：

```bash
python -m pip install -e .
```

可选功能：

```bash
python -m pip install -e ".[symbolic]"  # LaTeX→SymPy
python -m pip install -e ".[ml]"        # CRF切分器，需自行训练可信模型
python -m pip install -e ".[all,dev]"   # 全部功能与测试工具
```

当前仅在 GitHub 发布源码，尚未发布到 PyPI。

## 快速开始

```python
from math_clarify import analyze_formula, clarify_formula

result = analyze_formula(r"\lim_{x\to0}\frac{\sin x}{x}")
print(result.normalized_formula)
print(result.tokens)

ambiguous = clarify_formula("|A|=1")
print(ambiguous.needs_clarification)  # True
print(ambiguous.message)              # 询问竖线的含义
```

命令行：

```bash
math-clarify "|A|=1"
math-clarify "\int_{0}^{1} x^2\,dx" --symbolic
```

公式检索与融合：

```python
from math_clarify import FormulaDocument, dual_channel_rank

documents = [
    FormulaDocument("intro", "函数极限简介"),
    FormulaDocument("alpha", "f(x)=A+alpha(x), alpha(x)->0"),
]
dense_ids = ["intro", "alpha"]  # 由调用方提供，包本身不调用Embedding
results = dual_channel_rank("f(x)=A+α(x)，α(x)→0", documents, dense_ids)
print(results[0]["document_id"])
```

## 包含什么

- **规范化**：统一部分Unicode/LaTeX符号、箭头、希腊字母、简单分数与矩阵写法；供检索使用。
- **歧义反问**：用确定性规则发现常见多义记号；可以接收外部切分器的低置信度信号。
- **公式检索**：公式词项BM25和RRF融合；调用方提供自然语言通道排序。
- **可选符号解析**：用 `latex2sympy2-extended` 转为SymPy结构，保留积分变量和上下限等字段；不会自动求值。
- **可选CRF代码**：字符级公式切分与角色识别的训练/推断实现；仓库不附预训练模型。

## 重要限制

规范化并不保证数学等价：它会把 `A` 和 `a` 都转成小写检索词，可能合并数学上不同的对象。需要区分大小写时，应同时保留原公式，或使用可选符号解析通道。CRF切分器在先前的合成模板数据上存在泛化不足，因此没有把模型权重和合成成绩作为通用性能承诺。这里的歧义提示是候选问题，仍需结合题目上下文。

本仓库不包含教材PDF、页面截图、教材转录、个人路径、训练数据、模型权重或RAG02私有索引，也不需要云端API。

## 测试

```bash
python -m pip install -e ".[all,dev]"
python -m pytest
python -m build
```

## 许可

源码按 [MIT许可证](LICENSE) 提供。第三方依赖遵循各自许可证；请勿把原教材扫描件或转录内容随本包再分发。

