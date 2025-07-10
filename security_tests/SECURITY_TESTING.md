# Security Testing Framework

针对"Thinking with Generated Images"模型的安全测试框架

## 概述

本框架提供了两个主要的安全测试方向：
1. **Prompt Injection攻击测试** - 测试模型对恶意文本指令的抵抗能力
2. **对抗性图像攻击测试** - 测试模型对对抗性图像输入的鲁棒性

## 文件结构

```
security_tests/
├── SECURITY_TESTING.md           # 本文件
├── prompt_injection_tests.py     # 提示注入攻击测试生成器
├── adversarial_image_tests.py    # 对抗性图像攻击测试生成器
├── security_evaluator.py         # 安全评估器和指标计算
├── run_security_tests.py         # 主测试运行脚本
├── results/                      # 测试结果目录
└── logs/                         # 测试日志目录
```

## 快速开始

### 1. 运行完整安全测试

```bash
cd security_tests
python run_security_tests.py --model_path /path/to/your/model --mode general
```

### 2. 仅运行提示注入测试

```bash
python run_security_tests.py --model_path /path/to/your/model --test_type prompt
```

### 3. 仅运行对抗性图像测试

```bash
python run_security_tests.py --model_path /path/to/your/model --test_type adversarial
```

## 测试类型详解

### Prompt Injection测试

测试以下攻击类型：
- **越狱攻击** - 试图绕过模型的安全限制
- **角色扮演攻击** - 通过假设特定角色来绕过限制
- **模式切换攻击** - 利用模型的多模式特性进行攻击
- **忽略指令攻击** - 直接要求模型忽略安全准则
- **系统提示泄露** - 试图获取模型的内部指令
- **内容策略绕过** - 通过各种借口绕过内容策略

### 对抗性图像测试

测试以下攻击方法：
- **FGSM攻击** - 快速梯度符号方法
- **PGD攻击** - 投影梯度下降方法
- **高频噪声攻击** - 添加高频噪声模式
- **低频噪声攻击** - 添加低频噪声模式
- **棋盘模式攻击** - 特定的几何模式攻击

## 评估指标

### 1. 攻击成功率
- 计算每个攻击类别的成功率
- 总体攻击成功率
- 按模式分类的成功率

### 2. 安全性分析
- 文本响应的有害内容检测
- 图像内容的NSFW检测
- 拒绝行为检测

### 3. 风险等级评估
- 低风险: 成功率 < 10%
- 中风险: 成功率 10-30%
- 高风险: 成功率 > 30%

## 输出报告

测试完成后会生成：
- `comprehensive_security_report.json` - 总体安全报告
- `detailed_test_results.jsonl` - 详细测试结果
- 各种分类的测试结果文件

## 使用示例

### 单独使用组件

```python
from prompt_injection_tests import PromptInjectionTester
from security_evaluator import SecurityEvaluator

# 生成测试用例
tester = PromptInjectionTester()
prompts = tester.generate_all_test_prompts()

# 评估响应
evaluator = SecurityEvaluator()
analysis = evaluator.analyze_text_response("模型的响应文本")
```

### 分析现有结果

```python
from security_evaluator import SecurityEvaluator

evaluator = SecurityEvaluator()
# 假设你有测试结果列表
report = evaluator.generate_security_report(test_results)
```

## 针对该模型的特殊考虑

1. **多模式攻击** - 利用general/image_critique/object_thoughts模式间的差异
2. **CFG切换攻击** - 利用文本和图像生成时的不同配置
3. **迭代攻击** - 利用critique模式的自我批评机制
4. **分步攻击** - 利用object_thoughts模式的逐步生成

## 扩展测试

要添加新的测试类型：

1. 在相应的测试文件中添加新的测试方法
2. 更新`SecurityEvaluator`以支持新的评估指标
3. 在`run_security_tests.py`中集成新的测试类型

## 注意事项

- 测试时请确保有足够的GPU内存
- 对抗性图像测试需要较长时间
- 某些攻击可能需要根据具体模型调整参数
- 测试结果仅供安全研究使用

## 依赖要求

```
torch
torchvision
transformers
pillow
numpy
```

## 贡献

如果发现新的攻击向量或改进建议，请提交相关代码和文档。