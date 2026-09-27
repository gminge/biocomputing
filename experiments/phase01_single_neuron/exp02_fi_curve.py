"""实验 1.2：f-I 曲线 —— 发放频率随输入电流的变化。

对每个恒定电流强度做一次模拟，统计平均发放率 (Hz)，
并与解析解对比：

    f(I) = 1000 / [ t_ref + τ_m · ln((V_ss - V_reset) / (V_ss - V_th)) ]
    其中 V_ss = E_L + R_m·I，仅当 V_ss > V_th 时发放。

这是单神经元最重要的"输入-输出"特性：它把连续电流编码成脉冲频率。

运行：
    python experiments/phase01_single_neuron/exp02_fi_curve.py
输出：
    figures/exp02_fi_curve.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

import bclab.plotting  # noqa: F401  引入统一字体配置
from bclab import LIFNeuron

DT = 0.1            # ms
DURATION = 1000.0   # 每个电流点模拟 1 秒

FIG_DIR = Path(__file__).parent / "figures"


def main() -> None:
    neuron = LIFNeuron()
    p = neuron.params
    i_th = (p.v_th - p.e_l) / p.r_m

    currents = np.linspace(0.0, 2.5 * i_th, 30)
    rates_sim = np.array([
        neuron.simulate(I, dt=DT, duration=DURATION).firing_rate()
        for I in currents
    ])
    rates_theory = np.array([neuron.analytic_rate(I) for I in currents])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(currents, rates_sim, "o", color="tab:blue", ms=5,
            label="simulation")
    ax.plot(currents, rates_theory, "-", color="tab:red", lw=1.5,
            label="analytic")
    ax.axvline(i_th, ls=":", color="gray",
               label=f"rheobase I_th = {i_th:.2f} nA")

    ax.set_xlabel("input current I (nA)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("f-I curve of the LIF neuron")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "exp02_fi_curve.png"
    fig.savefig(out, dpi=150)

    max_err = np.abs(rates_sim - rates_theory).max()
    print(f"max |sim - analytic| = {max_err:.2f} Hz over {len(currents)} points")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
