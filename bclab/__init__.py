"""Biocomputing Lab —— 模拟神经元计算的个人实验室。

按阶段逐步演进：
    Phase 1  单个神经元（LIF 模型）
    Phase 2  突触与神经元连接
    Phase 3  脉冲神经网络
    Phase 4  STDP 可塑性学习
    Phase 5  简单计算任务
    Phase 6  复杂神经元模型（Hodgkin-Huxley 等）
    Phase 7  神经形态计算
    Phase 8  真实生物神经元计算
"""

from bclab.neurons import LIFNeuron, LIFParams, SimResult

__all__ = ["LIFNeuron", "LIFParams", "SimResult"]
__version__ = "0.1.0"
