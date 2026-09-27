"""实验 1.1：阶跃电流刺激 —— 观察膜电位的积累与发放。

给 LIF 神经元注入一段方波电流（前 20ms 为 0，随后跳到固定值），
对比两种情况：
    (a) 阈下刺激：电流太弱，V  asymptotically 趋近稳态但永远达不到阈值；
    (b) 阈上刺激：V 冲到阈值 → 发放 → 复位 → 不应期 → 再次充电，周期性发放。

运行：
    python experiments/phase01_single_neuron/exp01_step_current.py
输出：
    figures/exp01_step_current.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from bclab import LIFNeuron
from bclab.plotting import plot_voltage_trace

DT = 0.1          # ms
T_PRE = 20.0      # 刺激前的基线时长 (ms)
T_STIM = 180.0    # 刺激时长 (ms)

FIG_DIR = Path(__file__).parent / "figures"


def step_current(amplitude: float) -> np.ndarray:
    """前 T_PRE 毫秒为 0，之后恒为 amplitude 的电流波形。"""
    n_pre = int(round(T_PRE / DT))
    n_stim = int(round(T_STIM / DT))
    return np.concatenate([np.zeros(n_pre), np.full(n_stim, amplitude)])


def main() -> None:
    neuron = LIFNeuron()  # 默认参数：R_m=10 MΩ，V_th=-50 mV，E_L=-65 mV
    p = neuron.params

    # 临界电流：恰好把稳态电位推到阈值的电流 I_th = (V_th - E_L)/R_m
    i_th = (p.v_th - p.e_l) / p.r_m
    print(f"临界电流 I_th = {i_th:.2f} nA（稳态电位刚好等于阈值）")

    cases = [
        (0.95 * i_th, "(a) 阈下刺激：永不发放"),
        (1.60 * i_th, "(b) 阈上刺激：周期性发放"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    for ax, (amp, title) in zip(axes, cases):
        result = neuron.simulate(step_current(amp), dt=DT)
        plot_voltage_trace(result, ax=ax, title=f"{title}  I={amp:.2f} nA")
        ax.annotate(f"{result.n_spikes} spikes", xy=(0.02, 0.05),
                    xycoords="axes fraction", fontsize=9, color="tab:red")
        print(f"I = {amp:.2f} nA -> {result.n_spikes} spikes, "
              f"rate = {result.firing_rate():.1f} Hz")

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "exp01_step_current.png"
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
