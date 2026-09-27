"""神经元模型。

Phase 1 实现最经典的 LIF（Leaky Integrate-and-Fire，漏电整合发放）模型：

    τ_m · dV/dt = (E_L - V) + R_m · I(t)     当 V < V_th
    V ← V_reset                              当 V ≥ V_th（发放一个脉冲）
    随后 t_ref 毫秒内 V 固定在 V_reset（不应期）

其中：
    V     膜电位 (mV)
    τ_m   膜时间常数 (ms)，决定电位"泄漏"的速度
    E_L   静息（漏）电位 (mV)
    R_m   膜电阻 (MΩ)，R_m·I 是电流能抬升的电位幅度
    V_th  发放阈值 (mV)
    t_ref 绝对不应期 (ms)

后续 Phase 6 会在本文件中加入 Hodgkin-Huxley 等更复杂的模型，
保持同样的 simulate() 接口以便对比。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LIFParams:
    """LIF 神经元的全部参数，默认值取自常见教科书设置。"""

    tau_m: float = 20.0      # 膜时间常数 (ms)
    e_l: float = -65.0       # 静息电位 (mV)
    v_reset: float = -65.0   # 复位电位 (mV)
    v_th: float = -50.0      # 发放阈值 (mV)
    r_m: float = 10.0        # 膜电阻 (MΩ)，I 单位 nA 时 R·I 得 mV
    t_ref: float = 2.0       # 绝对不应期 (ms)


@dataclass
class SimResult:
    """一次模拟的输出。"""

    t: np.ndarray            # 时间轴 (ms)，shape (n_steps,)
    v: np.ndarray            # 膜电位轨迹 (mV)，shape (n_steps,)
    spikes: np.ndarray       # 发放时刻 (ms)，shape (n_spikes,)
    dt: float                # 积分步长 (ms)
    params: LIFParams = field(default_factory=LIFParams)

    @property
    def n_spikes(self) -> int:
        return len(self.spikes)

    @property
    def duration(self) -> float:
        """模拟总时长 (ms)。"""
        return float(self.t[-1] + self.dt)

    def firing_rate(self) -> float:
        """平均发放率 (Hz)。"""
        return self.n_spikes / (self.duration / 1000.0)


class LIFNeuron:
    """LIF 神经元，用显式欧拉法逐步积分。

    用法::

        neuron = LIFNeuron()                    # 默认参数
        neuron = LIFNeuron(v_th=-52.0)          # 覆盖任意 LIFParams 字段
        result = neuron.simulate(current, dt)   # current: 每步输入电流 (nA)

    输入电流可以是：
        - np.ndarray，长度决定模拟步数；
        - 标量，表示恒定电流，此时必须给定 duration。
    """

    def __init__(self, **overrides):
        self.params = LIFParams(**overrides)

    # ------------------------------------------------------------------
    def simulate(
        self,
        current: np.ndarray | float,
        dt: float = 0.1,
        duration: float | None = None,
    ) -> SimResult:
        """积分整个时间进程，返回 SimResult。

        current : 每个时间步的输入电流 (nA)，或标量恒定电流
        dt      : 积分步长 (ms)
        duration: 当 current 为标量时的模拟时长 (ms)
        """
        p = self.params

        if np.isscalar(current):
            if duration is None:
                raise ValueError("恒定电流输入时必须指定 duration (ms)")
            current = np.full(int(round(duration / dt)), float(current))
        else:
            current = np.asarray(current, dtype=float)

        n_steps = len(current)
        t = np.arange(n_steps) * dt
        v = np.empty(n_steps)
        spikes = []

        v_now = p.e_l
        ref_steps_left = 0          # 剩余不应期步数
        ref_steps = int(round(p.t_ref / dt))

        for i in range(n_steps):
            v[i] = v_now

            if ref_steps_left > 0:
                # 不应期内：电位钳在复位值，不积分
                ref_steps_left -= 1
                v_now = p.v_reset
                continue

            # 显式欧拉：V += dt * dV/dt
            v_now += dt / p.tau_m * (p.e_l - v_now + p.r_m * current[i])

            if v_now >= p.v_th:
                spikes.append(t[i])
                v_now = p.v_reset
                ref_steps_left = ref_steps

        return SimResult(
            t=t, v=v, spikes=np.array(spikes), dt=dt, params=p
        )

    # ------------------------------------------------------------------
    def analytic_rate(self, current: float) -> float:
        """恒定电流下发放率的解析解 (Hz)。

        由 V_ss = E_L + R_m·I 为稳态电位，若 V_ss <= V_th 则永不发放；
        否则从 V_reset 充电到 V_th 需时 τ_m·ln((V_ss-V_reset)/(V_ss-V_th))，
        加上不应期即为一个脉冲间隔。
        """
        p = self.params
        v_ss = p.e_l + p.r_m * current
        if v_ss <= p.v_th:
            return 0.0
        isi = p.t_ref + p.tau_m * np.log(
            (v_ss - p.v_reset) / (v_ss - p.v_th)
        )
        return 1000.0 / isi
